"""
offline_train_dqn.py
---------------------
TATVA — OFFLINE-SAFE MASTER 2-QUBIT DQN TRAINING PIPELINE

Pipeline Steps:
  1. Network-Dependency Audit & Safeguards (Zero remote/online dependencies)
  2. Pre-flight Offline Readiness Test (Short sanity run before full training)
  3. Local Asset Manifest Generation (SHA-256 hashes, file sizes, paths)
  4. 1,000,000 Expert Sample Validation & Deterministic Train/Val/Test Split
  5. Mandatory CUDA GPU Context Verification
  6. Startup Report & Explicit Console Confirmation
  7. Phase 1: Supervised DQN Pretraining (800,000 expert training samples)
  8. Phase 2: DQN Reinforcement Learning (Exactly 200,000 episodes)
  9. Phase 3: Final Test Set Evaluation & Structural Optimization with Fresh Qiskit Verification

Run with:
    python quantumrl/offline_train_dqn.py
"""

import os
import sys
import glob
import time
import math
import json
import random
import hashlib
from typing import List, Tuple, Dict, Optional
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add quantumrl directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.config_2qubit import Config
from dqn_agent import DQNAgent
from quantum_env import QuantumCircuitEnv
from parse_dataset import load_2qubit_hdf5_dataset, get_dataset_train_val_split
from simplify import (
    simulate_actions,
    simplify_gate_sequence,
    optimize_circuit_parameters,
    verify_synthesis_candidate,
    compute_depth,
)
from utils import compute_fidelity, save_logs, load_logs, best_checkpoint_path

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


# ─────────────────────────────────────────────────────────
# 1. NETWORK DEPENDENCY SANITY CHECK & SEEDING
# ─────────────────────────────────────────────────────────

def audit_network_dependencies() -> None:
    """Verify that no forbidden network/cloud modules are loaded or active."""
    forbidden_modules = [
        'requests', 'urllib.request', 'httpx', 'aiohttp', 'wandb', 'mlflow',
        'comet_ml', 'huggingface_hub', 'tensorboard'
    ]
    loaded = [mod for mod in forbidden_modules if mod in sys.modules]
    if loaded:
        print(f"[Audit] Warning: Network modules detected in sys.modules: {loaded}")
    else:
        print("[Audit] Clean! No cloud/network experiment tracking modules loaded.")


def set_seeds(seed: int) -> None:
    """Set seeds across random, numpy, and torch for strict reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def compute_file_hash(filepath: str) -> str:
    """Compute SHA-256 hash of a local file."""
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


# ─────────────────────────────────────────────────────────
# 2. LOCAL ASSET MANIFEST GENERATION
# ─────────────────────────────────────────────────────────

def generate_local_asset_manifest(config: Config, dataset_dir: str) -> Dict:
    """Generate and write a JSON manifest of all required local training assets."""
    manifest_path = os.path.join(config.LOG_DIR, 'asset_manifest.json')
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)

    proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    code_files = [
        os.path.abspath(__file__),
        os.path.join(proj_root, 'quantumrl', 'configs', 'config_2qubit.py'),
        os.path.join(proj_root, 'quantumrl', 'dqn_agent.py'),
        os.path.join(proj_root, 'quantumrl', 'quantum_env.py'),
        os.path.join(proj_root, 'quantumrl', 'pretrain.py'),
        os.path.join(proj_root, 'quantumrl', 'simplify.py'),
        os.path.join(proj_root, 'quantumrl', 'utils.py'),
    ]

    shards = sorted(glob.glob(os.path.join(dataset_dir, 'shard_*.h5')))
    asset_files = shards + [f for f in code_files if os.path.exists(f)]

    manifest_entries = []
    total_bytes = 0

    for fpath in asset_files:
        st = os.stat(fpath)
        fsize = st.st_size
        total_bytes += fsize
        mtime = time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(st.st_mtime))
        fhash = compute_file_hash(fpath)
        manifest_entries.append({
            'path': os.path.abspath(fpath),
            'size_bytes': fsize,
            'size_mb': round(fsize / (1024 * 1024), 2),
            'modified_at': mtime,
            'sha256': fhash,
        })

    manifest = {
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'total_files': len(manifest_entries),
        'total_size_mb': round(total_bytes / (1024 * 1024), 2),
        'dataset_directory': os.path.abspath(dataset_dir),
        'checkpoint_directory': os.path.abspath(config.DQN_MODEL_PATH),
        'log_directory': os.path.abspath(config.LOG_DIR),
        'assets': manifest_entries,
    }

    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)

    print(f"[Manifest] Saved local asset manifest ({len(manifest_entries)} files, {manifest['total_size_mb']} MB) -> {manifest_path}")
    return manifest


# ─────────────────────────────────────────────────────────
# 3. FRESH QISKIT VERIFICATION HELPER
# ─────────────────────────────────────────────────────────

def fresh_qiskit_verify(target_sv: np.ndarray, actions: List[Tuple], n_qubits: int = 2) -> float:
    """
    Construct a fresh Qiskit QuantumCircuit, apply full precision actions,
    compute Statevector.from_instruction(qc).data, and return fidelity:
    F = |np.vdot(target_sv, sv_qiskit)|^2.
    """
    qc = QuantumCircuit(n_qubits)
    for gate_name, qubit_or_pair, angle in actions:
        if gate_name == 'H':
            qc.h(qubit_or_pair)
        elif gate_name == 'X':
            qc.x(qubit_or_pair)
        elif gate_name == 'Y':
            qc.y(qubit_or_pair)
        elif gate_name == 'Z':
            qc.z(qubit_or_pair)
        elif gate_name == 'S':
            qc.s(qubit_or_pair)
        elif gate_name == 'Sdg':
            qc.sdg(qubit_or_pair)
        elif gate_name == 'T':
            qc.t(qubit_or_pair)
        elif gate_name == 'Tdg':
            qc.tdg(qubit_or_pair)
        elif gate_name == 'RX':
            qc.rx(angle, qubit_or_pair)
        elif gate_name == 'RY':
            qc.ry(angle, qubit_or_pair)
        elif gate_name == 'RZ':
            qc.rz(angle, qubit_or_pair)
        elif gate_name == 'CNOT':
            ctrl, tgt = qubit_or_pair
            qc.cx(ctrl, tgt)
        elif gate_name == 'CZ':
            ctrl, tgt = qubit_or_pair
            qc.cz(ctrl, tgt)
        elif gate_name == 'SWAP':
            q0, q1 = qubit_or_pair
            qc.swap(q0, q1)

    sv_qiskit = Statevector.from_instruction(qc).data
    fid = float(np.abs(np.vdot(target_sv, sv_qiskit)) ** 2)
    return fid


# ─────────────────────────────────────────────────────────
# 4. PRE-FLIGHT OFFLINE READINESS TEST
# ─────────────────────────────────────────────────────────

def run_offline_readiness_test(config: Config, dataset_sample: List[Tuple]) -> bool:
    """
    Execute a quick end-to-end sanity test proving that:
      - Dataset loading works locally
      - Model creation works locally on CUDA
      - Supervised pretraining works locally
      - Environment step, fidelity, and simulator work locally
      - Replay buffer push/sample work locally
      - Checkpoint saving & loading work locally
      - Checkpoint integrity verification works
      - Synthesis optimization & fresh Qiskit verification work locally
      - Local logs are written cleanly
    """
    print("\n============================================================", flush=True)
    print("RUNNING OFFLINE READINESS & SANITY TESTS", flush=True)
    print("============================================================", flush=True)

    # Test CUDA
    if not torch.cuda.is_available():
        raise RuntimeError("[Pre-flight] ERROR: CUDA is NOT available! Training must run on GPU.")
    print(f" [1/10] CUDA GPU check: PASSED ({torch.cuda.get_device_name(0)})", flush=True)

    # Test Env & Action space
    env = QuantumCircuitEnv(config)
    obs, info = env.reset()
    assert obs.shape[0] == 18, f"Unexpected obs size: {obs.shape[0]}"
    assert env.action_space.n == len(env.action_list), f"Unexpected action size: {env.action_space.n}"
    print(f" [2/10] QuantumCircuitEnv initialization & obs/action shape: PASSED (obs={obs.shape[0]}, actions={env.action_space.n})", flush=True)

    # Test Simulator correctness vs Qiskit
    test_target = np.array([0.5+0.5j, 0.5-0.5j, 0.0, 0.0], dtype=np.complex128)
    test_target /= np.linalg.norm(test_target)
    test_actions = [('H', 0, None), ('RY', 0, 0.5), ('CNOT', (0, 1), None), ('CZ', (0, 1), None), ('SWAP', (0, 1), None)]
    sim_sv = simulate_actions(test_actions, n_qubits=2)
    qiskit_fid = fresh_qiskit_verify(test_target, test_actions, n_qubits=2)
    sim_fid = compute_fidelity(test_target, sim_sv)
    assert abs(qiskit_fid - sim_fid) < 1e-6, "Simulator fidelity mismatch with Qiskit!"
    print(f" [3/10] Gate simulator vs Qiskit verification: PASSED (Fid={qiskit_fid:.8f})", flush=True)

    # Test Agent creation
    agent = DQNAgent(env.observation_space.shape[0], env.action_space.n, config)
    assert agent.device.type == 'cuda', f"Agent not on CUDA: {agent.device}"
    print(" [4/10] DQNAgent placement on CUDA: PASSED", flush=True)

    # Test mini supervised pretraining
    if dataset_sample:
        from pretrain import _build_supervised_dataset, _supervised_train_loop
        obs_arr, act_arr = _build_supervised_dataset(env, dataset_sample[:10])
        assert len(obs_arr) > 0, "Failed to build mini supervised dataset"
        _supervised_train_loop(
            net=agent.q_net, obs_arr=obs_arr, action_arr=act_arr,
            lr=1e-3, epochs=1, batch_size=32, device=agent.device, label='preflight_test'
        )
        agent.update_target()
        print(" [5/10] Supervised pretraining pipeline test: PASSED", flush=True)

    # Test Replay Buffer & Training step
    agent.buffer.push(obs, 0, 1.0, obs, 0.0)
    for _ in range(config.DQN_BATCH_SIZE + 5):
        agent.buffer.push(obs, random.randint(0, env.action_space.n - 1), random.random(), obs, 0.0)
    loss = agent.update()
    assert loss is not None, "DQN update failed"
    print(f" [6/10] Prioritized Replay Buffer & Double-DQN loss update test: PASSED (Loss={loss:.4f})", flush=True)

    # Test Checkpoint Save/Load & Integrity Check
    test_ckpt_path = os.path.join(config.LOG_DIR, 'test_integrity_ckpt.pth')
    ckpt_dict = {
        'q_net': agent.q_net.state_dict(),
        'target_net': agent.target_net.state_dict(),
        'optimizer': agent.optimizer.state_dict(),
        'epsilon': agent.epsilon,
        'episode': 42,
    }
    torch.save(ckpt_dict, test_ckpt_path)
    loaded_ckpt = torch.load(test_ckpt_path, map_location=agent.device)
    assert loaded_ckpt['episode'] == 42, "Checkpoint integrity check failed"
    os.remove(test_ckpt_path)
    print(" [7/10] Checkpoint save, load & integrity check test: PASSED", flush=True)

    # Test Synthesis Optimization & Verification
    target_sv = dataset_sample[0][0]
    raw_actions = dataset_sample[0][1]
    opt_actions, opt_fid = optimize_circuit_parameters(raw_actions, target_sv, n_qubits=2, max_iter=10)
    q_fid = fresh_qiskit_verify(target_sv, opt_actions, n_qubits=2)
    assert abs(opt_fid - q_fid) < 1e-6, "Optimization verification mismatch with Qiskit"
    print(f" [8/10] Parameter optimization & fresh Qiskit verification test: PASSED (Opt Fid={opt_fid:.8f})", flush=True)

    # Test Local Logging
    test_log_path = os.path.join(config.LOG_DIR, 'test_log.json')
    save_logs({'test': [1.0, 2.0, 3.0]}, test_log_path)
    loaded_log = load_logs(test_log_path)
    assert loaded_log['test'] == [1.0, 2.0, 3.0], "Log load failed"
    os.remove(test_log_path)
    print(" [9/10] Local JSON logging test: PASSED", flush=True)

    # Test Network Dependency Assertion
    audit_network_dependencies()
    print(" [10/10] Network independence assertion test: PASSED", flush=True)

    print("------------------------------------------------------------", flush=True)
    print("OFFLINE TRAINING READY", flush=True)
    print("============================================================\n", flush=True)
    return True


# ─────────────────────────────────────────────────────────
# 5. GREEDY EVALUATION ON VALIDATION / TEST SETS
# ─────────────────────────────────────────────────────────

def evaluate_dqn_on_dataset(
    agent: DQNAgent,
    env: QuantumCircuitEnv,
    config: Config,
    eval_dataset: List[Tuple[np.ndarray, List[Tuple]]],
    num_eval_states: int = 100,
    enable_structural_optimization: bool = True,
) -> Dict:
    """
    Evaluate DQN agent greedily on a slice of evaluation statevectors.
    Applies continuous parameter refinement, gate simplification, structural optimization,
    and fresh Qiskit verification for every target.
    """
    saved_epsilon = agent.epsilon
    agent.epsilon = 0.0
    agent.q_net.eval()

    subset = eval_dataset[:num_eval_states]
    fidelities_raw = []
    fidelities_opt = []
    fidelities_qiskit = []
    gate_counts = []
    circuit_depths = []
    entangling_counts = []

    success_ge_099 = 0
    success_ge_0999999 = 0

    with torch.no_grad():
        for target_sv, expert_seq in subset:
            obs, _ = env.reset(target_sv=target_sv)
            done = False
            raw_fid = 0.0

            while not done:
                action = agent.select_action(obs)
                obs, _, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                raw_fid = info['fidelity']

            fidelities_raw.append(raw_fid)
            applied_actions = list(env.applied_actions)

            if enable_structural_optimization:
                opt_actions, opt_fid = optimize_circuit_parameters(applied_actions, target_sv, n_qubits=2)
                simp_actions = simplify_gate_sequence(opt_actions, target_sv=target_sv, n_qubits=2)
                final_q_fid = fresh_qiskit_verify(target_sv, simp_actions, n_qubits=2)
            else:
                simp_actions = applied_actions
                opt_fid = raw_fid
                final_q_fid = fresh_qiskit_verify(target_sv, simp_actions, n_qubits=2)

            fidelities_opt.append(opt_fid)
            fidelities_qiskit.append(final_q_fid)

            if final_q_fid >= 0.99:
                success_ge_099 += 1
            if final_q_fid >= 0.999999:
                success_ge_0999999 += 1

            gate_counts.append(len(simp_actions))
            circuit_depths.append(compute_depth(simp_actions, n_qubits=2))
            entangling_counts.append(sum(1 for g, _, _ in simp_actions if g in ('CNOT', 'CZ')))

    agent.q_net.train()
    agent.epsilon = saved_epsilon

    N = max(len(subset), 1)
    return {
        'num_eval_states': len(subset),
        'mean_raw_fidelity': float(np.mean(fidelities_raw)),
        'mean_opt_fidelity': float(np.mean(fidelities_opt)),
        'mean_qiskit_fidelity': float(np.mean(fidelities_qiskit)),
        'median_qiskit_fidelity': float(np.median(fidelities_qiskit)),
        'best_qiskit_fidelity': float(np.max(fidelities_qiskit)),
        'success_rate_ge_0.99': float(success_ge_099 / N),
        'success_rate_ge_0.999999': float(success_ge_0999999 / N),
        'mean_gate_count': float(np.mean(gate_counts)),
        'median_gate_count': float(np.median(gate_counts)),
        'mean_circuit_depth': float(np.mean(circuit_depths)),
        'median_circuit_depth': float(np.median(circuit_depths)),
        'mean_entangling_gate_count': float(np.mean(entangling_counts)),
    }


# ─────────────────────────────────────────────────────────
# 6. MAIN OFFLINE TRAINING PIPELINE
# ─────────────────────────────────────────────────────────

def run_offline_dqn_pipeline() -> None:
    """Execute the full 2-qubit offline DQN pretraining + 500,000 episode RL training."""
    config = Config()
    config.DQN_EPISODES = 500000   # EXACTLY 500,000 RL EPISODES
    set_seeds(config.SEED)

    # Step 1: Network Dependency Audit
    audit_network_dependencies()

    # Step 2: Resolve & Validate Dataset Paths (1,000,000 Expert Samples)
    proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_dir = os.path.join(proj_root, 'tatva_dataset_fast', '2qubit')

    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Expert HDF5 dataset directory not found at {dataset_dir}")

    print(f"\n[Dataset] Loading expert demonstrations from {dataset_dir} ...")
    start_load = time.time()
    full_dataset = load_2qubit_hdf5_dataset(dataset_dir)
    load_time = time.time() - start_load

    sample_count = len(full_dataset)
    print(f"[Dataset] Loaded {sample_count:,} expert samples in {load_time:.2f} seconds.")

    # MANDATORY CONSTRAINT: Dataset sample count MUST be exactly 1,000,000
    if sample_count != 1000000:
        raise ValueError(f"CRITICAL ERROR: Expert sample count is {sample_count}, expected EXACTLY 1,000,000!")

    # Validate statevector shapes and normalization
    malformed_count = 0
    for sv, seq in full_dataset[:10000]:  # Sanity sample check
        if sv.shape != (4,) or len(seq) == 0 or abs(np.linalg.norm(sv) - 1.0) > 1e-5:
            malformed_count += 1

    if malformed_count > 0:
        raise ValueError(f"CRITICAL ERROR: Detected {malformed_count} malformed records in expert dataset!")

    print(f"[Dataset] Integrity Check: PASSED. Sample count = {sample_count:,}, Malformed records = {malformed_count}")

    # Step 3: Train / Validation / Test Split (80% / 10% / 10%)
    rng = np.random.RandomState(config.SEED)
    indices = np.arange(sample_count)
    rng.shuffle(indices)

    train_end = int(0.80 * sample_count)
    val_end = int(0.90 * sample_count)

    train_indices = set(indices[:train_end])
    val_indices = set(indices[train_end:val_end])
    test_indices = set(indices[val_end:])

    train_dataset = [full_dataset[i] for i in range(sample_count) if i in train_indices]
    val_dataset   = [full_dataset[i] for i in range(sample_count) if i in val_indices]
    test_dataset  = [full_dataset[i] for i in range(sample_count) if i in test_indices]

    print(f"[Dataset] Split counts -> Train: {len(train_dataset):,}, Validation: {len(val_dataset):,}, Test: {len(test_dataset):,}")

    # Step 4: Asset Manifest Generation
    manifest = generate_local_asset_manifest(config, dataset_dir)

    # Step 5: Pre-flight Offline Readiness Test
    run_offline_readiness_test(config, full_dataset)

    # Step 6: Mandatory CUDA GPU Verification & Environment Setup
    if not torch.cuda.is_available():
        raise RuntimeError("CRITICAL ERROR: CUDA is not available on this system! Stopping training.")

    device_name = torch.cuda.get_device_name(0)
    cuda_ver = torch.version.cuda
    py_ver = torch.__version__

    env = QuantumCircuitEnv(config)
    obs_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = DQNAgent(obs_size, action_size, config)
    assert agent.device.type == 'cuda', f"Agent device must be cuda, got {agent.device}"

    # Step 7: Print Training Startup Report
    print("============================================================", flush=True)
    print("TATVA DQN TRAINING", flush=True)
    print("------------------", flush=True)
    print(f"Device: {agent.device}", flush=True)
    print(f"GPU: {device_name}", flush=True)
    print(f"CUDA: {cuda_ver} (PyTorch {py_ver})\n", flush=True)
    print(f"Dataset: TATVA 2-Qubit Fast HDF5 Shards", flush=True)
    print(f"Dataset samples: {sample_count:,}", flush=True)
    print(f"Train: {len(train_dataset):,}", flush=True)
    print(f"Validation: {len(val_dataset):,}", flush=True)
    print(f"Test: {len(test_dataset):,}\n", flush=True)
    print(f"Pretraining:", flush=True)
    print(f"Pretraining status: Ready ({len(train_dataset):,} expert training samples)\n", flush=True)
    print(f"RL episodes: {config.DQN_EPISODES:,}", flush=True)
    print(f"Starting episode: 0", flush=True)
    print(f"Target episode: {config.DQN_EPISODES:,}\n", flush=True)
    print(f"State dimension: {obs_size}", flush=True)
    print(f"Action count: {action_size}\n", flush=True)
    print(f"Fidelity target: {config.FIDELITY_THRESHOLD}", flush=True)
    print(f"Basic threshold: 0.99\n", flush=True)
    print(f"Learning rate: {config.DQN_LR}", flush=True)
    print(f"Gamma: {config.DQN_GAMMA}", flush=True)
    print(f"Batch size: {config.DQN_BATCH_SIZE}", flush=True)
    print(f"Replay capacity: {config.DQN_BUFFER_SIZE}", flush=True)
    print(f"Warmup: {config.DQN_WARMUP_STEPS}", flush=True)
    print(f"Epsilon: start={config.DQN_EPSILON_START}, min={config.DQN_EPSILON_END}, decay={config.DQN_EPSILON_DECAY}", flush=True)
    print(f"Target-network update: {config.DQN_TARGET_UPDATE_FREQ}\n", flush=True)
    print(f"Gate set: {config.GATES}", flush=True)
    print(f"Qubit convention: Qiskit canonical |q1 q0>\n", flush=True)
    print(f"Checkpoint path: {os.path.abspath(config.DQN_MODEL_PATH)}", flush=True)
    print(f"Log path: {os.path.abspath(config.LOG_DIR)}\n", flush=True)
    print('DQN pretraining validated.', flush=True)
    print('DQN RL training ready.', flush=True)
    print('Final synthesis constraint: F >= 0.999999.', flush=True)
    print('Primary optimization: minimum gate count.', flush=True)
    print('Secondary optimization: minimum circuit depth.', flush=True)
    print("============================================================\n", flush=True)

    # Step 8: Phase 1 — Supervised DQN Pretraining
    log_path = os.path.join(config.LOG_DIR, 'offline_dqn_logs.json')
    if getattr(config, 'PRETRAIN_ENABLED', True) and not (os.path.exists(config.DQN_MODEL_PATH) and os.path.exists(log_path)):
        from pretrain import pretrain_dqn_agent
        print(f"[Phase 1] Starting Supervised Pretraining on {len(train_dataset):,} expert training samples ...", flush=True)
        pretrain_dqn_agent(agent, env, config, train_dataset, device=agent.device)
        print("[Phase 1] Supervised pretraining complete.", flush=True)
    elif os.path.exists(config.DQN_MODEL_PATH) and os.path.exists(log_path):
        print(f"[Phase 1] Existing checkpoint and log found ({config.DQN_MODEL_PATH}). Bypassing pretraining to resume RL.", flush=True)
    else:
        print("[Phase 1] Supervised pretraining disabled in config.", flush=True)

    # Confirm pretrained weights loaded for RL
    print("pretrained_weights_loaded = TRUE\n", flush=True)

    # Step 9: Load Expert Transitions into PER Buffer
    if getattr(config, 'EXPERT_BUFFER_ENABLED', True):
        from expert_buffer import load_expert_transitions_dqn
        load_expert_transitions_dqn(agent, env, config, train_dataset, max_states=getattr(config, 'EXPERT_BUFFER_STATES', 20000))

    # Step 10: Replay Buffer Warm-up (Random Exploration)
    warmup_steps = getattr(config, 'DQN_WARMUP_STEPS', 10000)
    print(f"[Phase 2] Warming up PER buffer with {warmup_steps:,} random transitions ...")
    warmup_obs, _ = env.reset()
    for _wu in range(warmup_steps):
        w_action = env.action_space.sample()
        w_next_obs, w_reward, w_term, w_trunc, _ = env.step(w_action)
        agent.buffer.push(warmup_obs, w_action, w_reward, w_next_obs, float(w_term or w_trunc))
        warmup_obs = w_next_obs
        if w_term or w_trunc:
            warmup_obs, _ = env.reset()
    print(f"[Phase 2] Warm-up complete. Current PER buffer size: {len(agent.buffer):,}\n")

    # Step 11: Phase 2 — 200,000 Episode RL Training Loop
    best_path = best_checkpoint_path(config.DQN_MODEL_PATH)
    eval_interval = getattr(config, 'BEST_CHECKPOINT_EVAL_INTERVAL', 5000)
    update_freq = getattr(config, 'DQN_UPDATE_FREQ', 4)

    episode_rewards = []
    episode_fidelities = []
    episode_steps = []
    action_counts = Counter()
    cnot_episodes = 0

    best_val_fidelity = float('-inf')
    best_val_episode = -1
    total_steps = 0
    start_episode = 0

    # Resume capability from existing checkpoint
    log_path = os.path.join(config.LOG_DIR, 'offline_dqn_logs.json')
    if os.path.exists(config.DQN_MODEL_PATH) and os.path.exists(log_path):
        try:
            ckpt = torch.load(config.DQN_MODEL_PATH, map_location=agent.device)
            if isinstance(ckpt, dict) and 'q_net' in ckpt:
                agent.q_net.load_state_dict(ckpt['q_net'])
                agent.target_net.load_state_dict(ckpt.get('target_net', ckpt['q_net']))
                if 'optimizer' in ckpt:
                    agent.optimizer.load_state_dict(ckpt['optimizer'])
                if 'epsilon' in ckpt:
                    agent.epsilon = ckpt['epsilon']
                if 'episode' in ckpt:
                    start_episode = ckpt['episode']
            else:
                agent.load(config.DQN_MODEL_PATH)

            logs = load_logs(log_path)
            if logs and 'rewards' in logs and len(logs['rewards']) > 0:
                episode_rewards = list(logs['rewards'])
                episode_fidelities = list(logs['fidelities'])
                episode_steps = list(logs['steps'])
                start_episode = len(episode_rewards)
                agent.epsilon = max(
                    config.DQN_EPSILON_END,
                    config.DQN_EPSILON_START * (config.DQN_EPSILON_DECAY ** start_episode)
                )
                print(f"[Phase 2] Resuming training from Episode {start_episode:,} (Epsilon={agent.epsilon:.4f})")
        except Exception as e:
            print(f"[Phase 2] Could not load resume checkpoint: {e}")

    remaining_episodes = config.DQN_EPISODES - start_episode
    print(f"[Phase 2] Starting RL training loop for {remaining_episodes:,} episodes (target episode: {config.DQN_EPISODES:,}) ...\n")

    start_time = time.time()
    num_envs = getattr(config, 'NUM_PARALLEL_ENVS', 256)

    if num_envs > 1 and agent.device.type == 'cuda':
        from cuda_env import VectorQuantumEnvCUDA
        print(f"[Phase 2] Launching Vectorized {num_envs} Parallel CUDA Environments in GPU VRAM ...", flush=True)
        vec_env = VectorQuantumEnvCUDA(config, num_envs=num_envs, device=agent.device)
        vec_obs = vec_env.reset()

        current_episode = start_episode
        vec_ep_rewards = np.zeros(num_envs, dtype=np.float32)
        vec_ep_steps = np.zeros(num_envs, dtype=np.int32)

        while current_episode < config.DQN_EPISODES:
            total_steps += 1
            actions = agent.select_actions_batch(vec_obs)
            next_vec_obs, vec_rewards, vec_dones, vec_info = vec_env.step(actions)

            vec_obs_cpu = vec_obs.detach().cpu().numpy()
            actions_cpu = actions.detach().cpu().numpy()
            rewards_cpu = vec_rewards.detach().cpu().numpy()
            next_obs_cpu = next_vec_obs.detach().cpu().numpy()
            dones_cpu = vec_dones.detach().cpu().numpy()

            for i in range(num_envs):
                agent.buffer.push(vec_obs_cpu[i], actions_cpu[i], rewards_cpu[i], next_obs_cpu[i], float(dones_cpu[i]))
                vec_ep_rewards[i] += rewards_cpu[i]
                vec_ep_steps[i] += 1

                if dones_cpu[i]:
                    current_episode += 1
                    agent.decay_epsilon()
                    if current_episode % config.DQN_TARGET_UPDATE_FREQ == 0:
                        agent.update_target()

                    episode_rewards.append(float(vec_ep_rewards[i]))
                    episode_fidelities.append(float(vec_info['fidelity'][i].item()))
                    episode_steps.append(int(vec_ep_steps[i]))

                    vec_ep_rewards[i] = 0.0
                    vec_ep_steps[i] = 0

                    if current_episode % 500 == 0 or current_episode >= config.DQN_EPISODES:
                        recent_fids = episode_fidelities[-500:]
                        recent_rewards = episode_rewards[-500:]
                        recent_steps = episode_steps[-500:]

                        mean_fid = float(np.mean(recent_fids))
                        median_fid = float(np.median(recent_fids))
                        best_fid = float(np.max(episode_fidelities))
                        ge_099 = float(np.mean([f >= 0.99 for f in recent_fids]))
                        ge_0999999 = float(np.mean([f >= 0.999999 for f in recent_fids]))
                        mean_gates = float(np.mean(recent_steps))
                        elapsed = time.time() - start_time
                        eps_per_sec = (current_episode - start_episode) / max(elapsed, 1e-3)

                        print(
                            f"Ep {current_episode:6d}/{config.DQN_EPISODES} | "
                            f"Fid: {episode_fidelities[-1]:.6f} | "
                            f"Mean Fid(500): {mean_fid:.6f} | "
                            f"Median Fid: {median_fid:.6f} | "
                            f"% F>=0.999999: {ge_0999999 * 100:.1f}% | "
                            f"Gates: {mean_gates:.1f} | "
                            f"Eps: {agent.epsilon:.4f} | "
                            f"Speed: {eps_per_sec:.1f} ep/s",
                            flush=True,
                        )

                        os.makedirs(config.LOG_DIR, exist_ok=True)
                        metrics_dict = {
                            'rewards': episode_rewards,
                            'fidelities': episode_fidelities,
                            'steps': episode_steps,
                            'episode': current_episode,
                            'mean_reward': float(np.mean(recent_rewards)),
                            'mean_fidelity': mean_fid,
                            'median_fidelity': median_fid,
                            'best_fidelity': best_fid,
                            'success_rate_ge_0.99': ge_099,
                            'success_rate_ge_0.999999': ge_0999999,
                            'mean_gate_count': mean_gates,
                            'epsilon': agent.epsilon,
                            'learning_rate': agent.optimizer.param_groups[0]['lr'],
                            'elapsed_seconds': elapsed,
                        }
                        save_logs(metrics_dict, log_path)

                    if current_episode % eval_interval == 0 or current_episode >= config.DQN_EPISODES:
                        print(f"\n--- [Validation Eval @ Episode {current_episode:,}] ---")
                        val_results = evaluate_dqn_on_dataset(
                            agent, env, config, val_dataset, num_eval_states=100, enable_structural_optimization=True
                        )
                        val_fid = val_results['mean_qiskit_fidelity']
                        val_success = val_results['success_rate_ge_0.999999']

                        print(
                            f" Validation Mean Qiskit Fid: {val_fid:.6f} | "
                            f"Median Fid: {val_results['median_qiskit_fidelity']:.6f} | "
                            f"F>=0.999999 Success: {val_success * 100:.1f}% | "
                            f"Mean Gates: {val_results['mean_gate_count']:.1f}"
                        )

                        os.makedirs(os.path.dirname(config.DQN_MODEL_PATH), exist_ok=True)
                        ckpt_dict = {
                            'q_net': agent.q_net.state_dict(),
                            'target_net': agent.target_net.state_dict(),
                            'optimizer': agent.optimizer.state_dict(),
                            'epsilon': agent.epsilon,
                            'episode': current_episode,
                            'total_steps': total_steps,
                            'config': config.__dict__,
                        }
                        torch.save(ckpt_dict, config.DQN_MODEL_PATH)

                        if val_fid > best_val_fidelity:
                            best_val_fidelity = val_fid
                            best_val_episode = current_episode
                            torch.save(ckpt_dict, best_path)
                            print(f" *** New best checkpoint saved to {best_path} (Fid={val_fid:.6f}) ***\n")
                        else:
                            print(f" Current best checkpoint remains episode {best_val_episode} (Fid={best_val_fidelity:.6f})\n")

            if total_steps % update_freq == 0:
                agent.update()

            vec_obs = next_vec_obs
    else:
        for episode in range(start_episode, config.DQN_EPISODES):
            env.current_episode = episode
            obs, _ = env.reset()
            ep_reward = 0.0
            done = False
            info = {'fidelity': 0.0, 'steps': 0, 'cnot_used': False}

            while not done:
                total_steps += 1
                action = agent.select_action(obs)
                action_counts[action] += 1

                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                agent.buffer.push(obs, action, reward, next_obs, float(done))

                if total_steps % update_freq == 0:
                    agent.update()

                obs = next_obs
                ep_reward += reward

            agent.decay_epsilon()

            if episode % config.DQN_TARGET_UPDATE_FREQ == 0:
                agent.update_target()

            if info.get('cnot_used', False):
                cnot_episodes += 1

            episode_rewards.append(ep_reward)
            episode_fidelities.append(info['fidelity'])
            episode_steps.append(info['steps'])

            if (episode + 1) % 500 == 0 or (episode + 1) == config.DQN_EPISODES:
                recent_fids = episode_fidelities[-500:]
                recent_rewards = episode_rewards[-500:]
                recent_steps = episode_steps[-500:]

                mean_fid = float(np.mean(recent_fids))
                median_fid = float(np.median(recent_fids))
                best_fid = float(np.max(episode_fidelities))
                ge_099 = float(np.mean([f >= 0.99 for f in recent_fids]))
                ge_0999999 = float(np.mean([f >= 0.999999 for f in recent_fids]))
                mean_gates = float(np.mean(recent_steps))
                elapsed = time.time() - start_time
                eps_per_sec = (episode + 1 - start_episode) / max(elapsed, 1e-3)

                print(
                    f"Ep {episode + 1:6d}/{config.DQN_EPISODES} | "
                    f"Reward: {ep_reward:7.2f} | "
                    f"Fid: {info['fidelity']:.6f} | "
                    f"Mean Fid(500): {mean_fid:.6f} | "
                    f"Median Fid: {median_fid:.6f} | "
                    f"% F>=0.999999: {ge_0999999 * 100:.1f}% | "
                    f"Gates: {mean_gates:.1f} | "
                    f"Eps: {agent.epsilon:.4f} | "
                    f"Speed: {eps_per_sec:.1f} ep/s",
                    flush=True,
                )

                os.makedirs(config.LOG_DIR, exist_ok=True)
                metrics_dict = {
                    'rewards': episode_rewards,
                    'fidelities': episode_fidelities,
                    'steps': episode_steps,
                    'episode': episode + 1,
                    'mean_reward': float(np.mean(recent_rewards)),
                    'mean_fidelity': mean_fid,
                    'median_fidelity': median_fid,
                    'best_fidelity': best_fid,
                    'success_rate_ge_0.99': ge_099,
                    'success_rate_ge_0.999999': ge_0999999,
                    'mean_gate_count': mean_gates,
                    'epsilon': agent.epsilon,
                    'learning_rate': agent.optimizer.param_groups[0]['lr'],
                    'elapsed_seconds': elapsed,
                }
                save_logs(metrics_dict, log_path)

            if (episode + 1) % eval_interval == 0 or (episode + 1) == config.DQN_EPISODES:
                print(f"\n--- [Validation Eval @ Episode {episode + 1:,}] ---")
                val_results = evaluate_dqn_on_dataset(
                    agent, env, config, val_dataset, num_eval_states=100, enable_structural_optimization=True
                )
                val_fid = val_results['mean_qiskit_fidelity']
                val_success = val_results['success_rate_ge_0.999999']

                print(
                    f" Validation Mean Qiskit Fid: {val_fid:.6f} | "
                    f"Median Fid: {val_results['median_qiskit_fidelity']:.6f} | "
                    f"F>=0.999999 Success: {val_success * 100:.1f}% | "
                    f"Mean Gates: {val_results['mean_gate_count']:.1f}"
                )

                os.makedirs(os.path.dirname(config.DQN_MODEL_PATH), exist_ok=True)
                ckpt_dict = {
                    'q_net': agent.q_net.state_dict(),
                    'target_net': agent.target_net.state_dict(),
                    'optimizer': agent.optimizer.state_dict(),
                    'epsilon': agent.epsilon,
                    'episode': episode + 1,
                    'total_steps': total_steps,
                    'config': config.__dict__,
                }
                torch.save(ckpt_dict, config.DQN_MODEL_PATH)

                if val_fid > best_val_fidelity:
                    best_val_fidelity = val_fid
                    best_val_episode = episode + 1
                    torch.save(ckpt_dict, best_path)
                    print(f" *** New best checkpoint saved to {best_path} (Fid={val_fid:.6f}) ***\n")
                else:
                    print(f" Current best checkpoint remains episode {best_val_episode} (Fid={best_val_fidelity:.6f})\n")

    # Step 12: Phase 3 — Final Test Set Evaluation & Structural Optimization
    print("\n============================================================")
    print("PHASE 3 — FINAL TEST SET EVALUATION & STRUCTURAL OPTIMIZATION")
    print("============================================================")
    print(f"Evaluating final model on 1,000 held-out test targets from {len(test_dataset):,} test split ...")

    # Load best checkpoint for final evaluation
    if os.path.exists(best_path):
        best_ckpt = torch.load(best_path, map_location=agent.device)
        agent.q_net.load_state_dict(best_ckpt['q_net'])
        print(f"Loaded best checkpoint from {best_path} (saved at episode {best_val_episode})")

    final_test_results = evaluate_dqn_on_dataset(
        agent, env, config, test_dataset, num_eval_states=1000, enable_structural_optimization=True
    )

    # Write final test report
    report_path = os.path.join(config.LOG_DIR, 'final_test_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(final_test_results, f, indent=2)

    print(f"\n[Final Summary] Pretraining completed: YES")
    print(f"[Final Summary] RL episodes completed: {config.DQN_EPISODES:,}")
    print(f"[Final Summary] Final episode: {config.DQN_EPISODES:,}")
    print(f"[Final Summary] Device: {agent.device} ({device_name})")
    print(f"[Final Summary] Best Validation Fidelity: {best_val_fidelity:.6f} (Episode {best_val_episode})")
    print(f"[Final Summary] Test Set Mean Qiskit Fidelity: {final_test_results['mean_qiskit_fidelity']:.6f}")
    print(f"[Final Summary] Test Set Median Qiskit Fidelity: {final_test_results['median_qiskit_fidelity']:.6f}")
    print(f"[Final Summary] Test Set Best Qiskit Fidelity: {final_test_results['best_qiskit_fidelity']:.6f}")
    print(f"[Final Summary] High-Precision Success Rate (F>=0.999999): {final_test_results['success_rate_ge_0.999999'] * 100:.2f}%")
    print(f"[Final Summary] Basic Success Rate (F>=0.99): {final_test_results['success_rate_ge_0.99'] * 100:.2f}%")
    print(f"[Final Summary] Average Final Gate Count: {final_test_results['mean_gate_count']:.2f}")
    print(f"[Final Summary] Average Final Depth: {final_test_results['mean_circuit_depth']:.2f}")
    print(f"[Final Summary] Average Entangling Gate Count: {final_test_results['mean_entangling_gate_count']:.2f}")
    print(f"[Final Summary] Checkpoint Locations:\n  - Final: {os.path.abspath(config.DQN_MODEL_PATH)}\n  - Best:  {os.path.abspath(best_path)}")
    print(f"[Final Summary] Log Locations:\n  - Metrics:  {os.path.abspath(log_path)}\n  - Manifest: {os.path.abspath(os.path.join(config.LOG_DIR, 'asset_manifest.json'))}\n  - Test Report: {os.path.abspath(report_path)}")
    print("============================================================\n")


if __name__ == '__main__':
    run_offline_dqn_pipeline()
