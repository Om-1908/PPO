"""
config_2qubit_smoke.py
----------------------
Temporary diagnostic configuration dataclass for QuantumRL — 2-Qubit Smoke Test.
All hyperparameters match config_2qubit.py, except episode counts and paths are
scaled down for a fast ~500 episode sanity check.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    """Single source of truth for 2-Qubit Diagnostic Smoke Test."""

    # ──────────────────────────────────────────────
    # Environment (2-Qubit Configuration)
    # ──────────────────────────────────────────────
    NUM_QUBITS: int = 2           # Scaled to 2 Qubits
    MAX_STEPS: int = 20           # Maximum gates applied per episode
    FIDELITY_THRESHOLD: float = 0.99   # Target fidelity to declare success
    GATE_PENALTY: float = 0.005  # Per-step penalty

    # ──────────────────────────────────────────────
    # Gate set available to the agent
    # ──────────────────────────────────────────────
    GATES: List[str] = field(
        default_factory=lambda: ['H', 'X', 'Y', 'Z', 'RX', 'RY', 'RZ', 'CNOT']
    )

    # 24-angle grid: 16 positive (π/8 … 2π) + 8 negative (-π/8 … -π)
    ROTATION_ANGLES: List[float] = field(
        default_factory=lambda: (
            [k * np.pi / 8 for k in range(1, 17)]   # π/8 … 2π  (16 angles)
            + [-k * np.pi / 8 for k in range(1, 9)]  # -π/8 … -π  (8 angles)
        )
    )

    # ──────────────────────────────────────────────
    # DQN Hyperparameters (Smoke test scaled down)
    # ──────────────────────────────────────────────
    DQN_EPISODES: int = 500
    DQN_BATCH_SIZE: int = 512
    DQN_BUFFER_SIZE: int = 200000
    DQN_LR: float = 0.0003
    DQN_GAMMA: float = 0.995
    DQN_EPSILON_START: float = 1.0
    DQN_EPSILON_END: float = 0.02
    DQN_EPSILON_DECAY: float = 0.99985
    DQN_TARGET_UPDATE_FREQ: int = 5
    DQN_HIDDEN_SIZE: int = 768
    DQN_WARMUP_STEPS: int = 500

    # Prioritized Experience Replay (PER)
    PER_ALPHA: float = 0.6
    PER_BETA_START: float = 0.4
    PER_BETA_FRAMES: float = 50000

    # ──────────────────────────────────────────────
    # PPO Hyperparameters (Smoke test scaled down)
    # ──────────────────────────────────────────────
    PPO_EPISODES: int = 500
    PPO_ROLLOUT_STEPS: int = 1024      # Scaled rollout steps for fast smoke test
    PPO_EPOCHS: int = 15               # Gradient epochs per rollout update
    PPO_MINI_BATCH_SIZE: int = 512     # Minibatch size for smooth CUDA optimization
    PPO_LR: float = 3e-4
    PPO_GAMMA: float = 0.995
    PPO_GAE_LAMBDA: float = 0.95
    PPO_CLIP_EPSILON: float = 0.2
    PPO_ENTROPY_COEF: float = 0.01
    PPO_VALUE_COEF: float = 1.0
    PPO_MAX_GRAD_NORM: float = 0.5
    PPO_HIDDEN_SIZE: int = 768
    PPO_MODEL_PATH: str = 'saved_models/2qubit_smoke/ppo_model.pth'
    PPO_LOG_PATH: str = 'logs/2qubit_smoke/ppo_logs.json'
    PPO_PLOT_PATH: str = 'plots/2qubit_smoke/ppo_training.png'

    # LR scheduler for PPO
    PPO_LR_DECAY: bool = True
    PPO_LR_MIN: float = 1e-5

    # ──────────────────────────────────────────────
    # Evaluation
    # ──────────────────────────────────────────────
    NUM_TEST_STATES: int = 500
    SEED: int = 42

    # ──────────────────────────────────────────────
    # Best-checkpoint tracking
    # ──────────────────────────────────────────────
    BEST_CHECKPOINT_EVAL_INTERVAL: int = 100
    BEST_CHECKPOINT_EVAL_STATES: int = 50

    LOG_ACTION_HISTOGRAM: bool = True

    # Curriculum learning
    CURRICULUM_ENABLED: bool = False
    CURRICULUM_POOL_SIZE: int = 30

    # File paths (smoke-scoped)
    DQN_MODEL_PATH: str = 'saved_models/2qubit_smoke/dqn_model.pth'
    LOG_DIR: str = 'logs/2qubit_smoke/'
    PLOT_DIR: str = 'plots/2qubit_smoke/'

    def __post_init__(self):
        """Ensure qubit-scoped paths if {n} formatting is present."""
        if '{n}' in self.DQN_MODEL_PATH:
            self.DQN_MODEL_PATH = self.DQN_MODEL_PATH.format(n=self.NUM_QUBITS)
        if '{n}' in self.PPO_MODEL_PATH:
            self.PPO_MODEL_PATH = self.PPO_MODEL_PATH.format(n=self.NUM_QUBITS)
        if '{n}' in self.PPO_LOG_PATH:
            self.PPO_LOG_PATH = self.PPO_LOG_PATH.format(n=self.NUM_QUBITS)
        if '{n}' in self.PPO_PLOT_PATH:
            self.PPO_PLOT_PATH = self.PPO_PLOT_PATH.format(n=self.NUM_QUBITS)
        if '{n}' in self.LOG_DIR:
            self.LOG_DIR = self.LOG_DIR.format(n=self.NUM_QUBITS)
        if '{n}' in self.PLOT_DIR:
            self.PLOT_DIR = self.PLOT_DIR.format(n=self.NUM_QUBITS)
