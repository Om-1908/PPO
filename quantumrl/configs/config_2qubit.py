"""
config_2qubit.py
----------------
Central configuration dataclass for QuantumRL — Scaled to 2 Qubits.
All hyperparameters, paths, and environment settings live here.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    """Single source of truth for every hyperparameter in QuantumRL (2-Qubit Production)."""

    # ──────────────────────────────────────────────
    # Environment (2-Qubit Configuration)
    # ──────────────────────────────────────────────
    NUM_QUBITS: int = 2           # Scaled to 2 Qubits
    MAX_STEPS: int = 25           # Increased from 20: KAK decomposition needs up to 8 gates + margin
    FIDELITY_THRESHOLD: float = 0.999999   # Target fidelity to declare success (strict requirement)
    GATE_PENALTY: float = 0.005  # Per-step penalty

    # ──────────────────────────────────────────────
    # Gate set available to the agent
    # Extended: CZ, SWAP, S, Sdg, T, Tdg added for richer 2-qubit expressivity
    # Action space: 164 actions total (16 fixed 1Q + 144 rotation + 4 two-qubit)
    # ──────────────────────────────────────────────
    GATES: List[str] = field(
        default_factory=lambda: [
            'H', 'X', 'Y', 'Z', 'S', 'Sdg', 'T', 'Tdg',  # fixed 1Q (8 gates × 2 qubits = 16)
            'RX', 'RY', 'RZ',                               # rotation 1Q (3 × 24 × 2 = 144)
            'CNOT', 'CZ', 'SWAP',                           # 2-qubit (CNOT×2 + CZ×1 + SWAP×1 = 4)
        ]
    )

    # 96-angle grid: 64 positive (π/32 … 2π) + 32 negative (-π/32 … -π)
    # Resolution π/32 ≈ 5.625° gives max single-gate fidelity loss < 0.001
    # For 8-gate KAK sequence: cumulative max fidelity loss < 0.008 (≮ much better than 0.99 threshold)
    ROTATION_ANGLES: List[float] = field(
        default_factory=lambda: (
            [k * np.pi / 32 for k in range(1, 65)]   # π/32 … 2π  (64 angles)
            + [-k * np.pi / 32 for k in range(1, 33)]  # -π/32 … -π  (32 angles)
        )
    )

    # ──────────────────────────────────────────────
    # DQN Hyperparameters (2-Qubit Scaled + Dueling + PER)
    # ──────────────────────────────────────────────
    DQN_EPISODES: int = 500000    # EXACTLY 500,000 synthesis episodes
    DQN_BATCH_SIZE: int = 512
    DQN_BUFFER_SIZE: int = 200000
    DQN_LR: float = 0.0003
    DQN_GAMMA: float = 0.995
    DQN_EPSILON_START: float = 1.0
    DQN_EPSILON_END: float = 0.02
    DQN_EPSILON_DECAY: float = 0.999985
    DQN_TARGET_UPDATE_FREQ: int = 5
    DQN_HIDDEN_SIZE: int = 768
    DQN_WARMUP_STEPS: int = 10000
    DQN_UPDATE_FREQ: int = 4

    # Prioritized Experience Replay (PER)
    PER_ALPHA: float = 0.6
    PER_BETA_START: float = 0.4
    PER_BETA_FRAMES: float = 50000

    # ──────────────────────────────────────────────
    # PPO Hyperparameters (2-Qubit Scaled - Optimized)
    # ──────────────────────────────────────────────
    PPO_EPISODES: int = 500000    # EXACTLY 500,000 synthesis episodes
    PPO_ROLLOUT_STEPS: int = 4096
    PPO_EPOCHS: int = 10
    PPO_MINI_BATCH_SIZE: int = 512
    PPO_LR: float = 3e-4
    PPO_GAMMA: float = 0.995
    PPO_GAE_LAMBDA: float = 0.95
    PPO_CLIP_EPSILON: float = 0.2
    PPO_ENTROPY_COEF: float = 0.02       # Increased for 164-action space exploration
    PPO_VALUE_COEF: float = 1.0
    PPO_MAX_GRAD_NORM: float = 0.5
    PPO_HIDDEN_SIZE: int = 768
    PPO_MODEL_PATH: str = 'saved_models/2qubit/ppo_model.pth'
    PPO_LOG_PATH: str = 'logs/2qubit/ppo_logs.json'
    PPO_PLOT_PATH: str = 'plots/2qubit/ppo_training.png'

    # LR scheduler for PPO
    PPO_LR_DECAY: bool = True
    PPO_LR_MIN: float = 1e-5

    # ──────────────────────────────────────────────
    # Pretraining (supervised warm-start on FULL Haar dataset — 100,000 states)
    # Primary learning mechanism: 100k × 8 gates = 800k (obs,action) pairs per epoch
    # ──────────────────────────────────────────────
    PRETRAIN_ENABLED: bool = True
    PRETRAIN_DATASET_PATH: str = 'haar_2qubit_synthesis.md'   # relative to project root
    PRETRAIN_MAX_STATES: int = 100000     # Use full dataset
    PRETRAIN_EPOCHS: int = 15             # 15 epochs × 800k pairs
    PRETRAIN_LR: float = 3e-4            # Adam LR matched to RL training
    PRETRAIN_BATCH_SIZE: int = 512        # Large batch for efficient training
    # Load expert transitions into the DQN PER buffer at episode start
    EXPERT_BUFFER_ENABLED: bool = True
    EXPERT_BUFFER_STATES: int = 20000     # 20k states × 8 gates = 160k expert transitions

    # ──────────────────────────────────────────────
    # CNOT forcing (2-qubit: ensures entangling gates appear early in training)
    # ──────────────────────────────────────────────
    CNOT_FORCE_UNTIL_EPISODE: int = 10000   # Force for first 10k episodes
    CNOT_FIRST_USE_BONUS: float = 1.5

    # ──────────────────────────────────────────────
    # Evaluation
    # ──────────────────────────────────────────────
    NUM_TEST_STATES: int = 1000
    SEED: int = 42

    # ──────────────────────────────────────────────
    # Best-checkpoint tracking
    # ──────────────────────────────────────────────
    BEST_CHECKPOINT_EVAL_INTERVAL: int = 500
    BEST_CHECKPOINT_EVAL_STATES: int = 50

    LOG_ACTION_HISTOGRAM: bool = True

    # Curriculum learning (disabled)
    CURRICULUM_ENABLED: bool = False
    CURRICULUM_POOL_SIZE: int = 30

    # File paths (qubit-scoped)
    DQN_MODEL_PATH: str = 'saved_models/2qubit/dqn_model.pth'
    LOG_DIR: str = 'logs/2qubit/'
    PLOT_DIR: str = 'plots/2qubit/'

    def __post_init__(self):
        """Ensure qubit-scoped paths if {n} formatting is present."""
        for attr in ('DQN_MODEL_PATH', 'PPO_MODEL_PATH', 'PPO_LOG_PATH',
                     'PPO_PLOT_PATH', 'LOG_DIR', 'PLOT_DIR'):
            val = getattr(self, attr)
            if '{n}' in val:
                setattr(self, attr, val.format(n=self.NUM_QUBITS))
