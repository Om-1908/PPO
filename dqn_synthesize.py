"""
dqn_synthesize.py
-----------------
Standalone Terminal Inference Script for TATVA DQN Quantum Circuit Synthesis.

Supports both 1-Qubit and 2-Qubit systems using trained DQN agents ONLY.

Usage:
------
  1. Interactive mode (run directly in terminal):
         python dqn_synthesize.py

  2. Command line interface (CLI mode):
         python dqn_synthesize.py --qubits 1 --target plus
         python dqn_synthesize.py --qubits 2 --target bell
         python dqn_synthesize.py --qubits 2 --target "0.5, 0.5, 0.5, 0.5"
         python dqn_synthesize.py --qubits 1 --target "1/sqrt(2), 1i/sqrt(2)"
         python dqn_synthesize.py --qubits 2 --target "-0.7989-0.4544i, +0.0354-0.2684i, -0.0138+0.1217i, -0.1021+0.2377i"

  3. Options:
         --budget N         Number of DQN policy candidate rollouts (default: 30)
         --fidelity F       Target fidelity threshold (default: 0.999 for 1Q, 0.999999 for 2Q)
"""

import argparse
import math
import os
import sys
import time
from typing import List, Optional, Tuple

import numpy as np
import torch

# Resolve quantumrl path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
QUANTUMRL_DIR = os.path.join(SCRIPT_DIR, 'quantumrl')
if QUANTUMRL_DIR not in sys.path:
    sys.path.insert(0, QUANTUMRL_DIR)

from configs.config_1qubit import Config as Config1Qubit
from configs.config_2qubit import Config as Config2Qubit
from quantum_env import QuantumCircuitEnv
from dqn_agent import DQNAgent
from utils import generate_random_statevector
from simplify import verify_synthesis_candidate, _fidelity
from synthesis import (
    build_qiskit_circuit,
    format_circuit_display_lines,
    prune_and_reoptimize_structural_search,
    verify_4way_consistency,
    candidate_rank_key,
)


# ─────────────────────────────────────────────────────────────────────────────
# Named Presets
# ─────────────────────────────────────────────────────────────────────────────

INV_SQRT2 = 1.0 / math.sqrt(2.0)

PRESETS_1Q = {
    'zero':    np.array([1.0+0j, 0.0+0j],          dtype=np.complex128),
    'one':     np.array([0.0+0j, 1.0+0j],          dtype=np.complex128),
    'plus':    np.array([INV_SQRT2, INV_SQRT2],    dtype=np.complex128),
    'minus':   np.array([INV_SQRT2, -INV_SQRT2],   dtype=np.complex128),
    'i_state': np.array([INV_SQRT2, INV_SQRT2*1j], dtype=np.complex128),
}

PRESETS_2Q = {
    'bell':      np.array([INV_SQRT2, 0.0, 0.0, INV_SQRT2],  dtype=np.complex128),
    'phi_plus':  np.array([INV_SQRT2, 0.0, 0.0, INV_SQRT2],  dtype=np.complex128),
    'phi_minus': np.array([INV_SQRT2, 0.0, 0.0, -INV_SQRT2], dtype=np.complex128),
    'psi_plus':  np.array([0.0, INV_SQRT2, INV_SQRT2, 0.0],  dtype=np.complex128),
    'psi_minus': np.array([0.0, INV_SQRT2, -INV_SQRT2, 0.0], dtype=np.complex128),
    'ghz':       np.array([INV_SQRT2, 0.0, 0.0, INV_SQRT2],  dtype=np.complex128),
    'zero':      np.array([1.0, 0.0, 0.0, 0.0],              dtype=np.complex128),
    'plus':      np.array([0.5, 0.5, 0.5, 0.5],              dtype=np.complex128),
}


# ─────────────────────────────────────────────────────────────────────────────
# Input Parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_statevector(input_str: str, n_qubits: Optional[int] = None) -> Tuple[np.ndarray, int]:
    """
    Parse a user input string into a normalized complex128 statevector.
    Supports: preset names, 'random', or custom comma-separated amplitudes.
    """
    clean = input_str.strip().lower()

    if clean == 'random':
        target_n = n_qubits if n_qubits in (1, 2) else 2
        return generate_random_statevector(target_n), target_n

    if n_qubits == 1 and clean in PRESETS_1Q:
        return PRESETS_1Q[clean], 1
    if n_qubits == 2 and clean in PRESETS_2Q:
        return PRESETS_2Q[clean], 2
    if n_qubits is None:
        if clean in PRESETS_1Q:
            return PRESETS_1Q[clean], 1
        if clean in PRESETS_2Q:
            return PRESETS_2Q[clean], 2

    # Custom amplitude parsing
    s = clean.replace('[', '').replace(']', '').replace('(', '').replace(')', '')
    s = s.replace('sqrt', 'np.sqrt').replace('pi', 'np.pi')
    s = s.replace('i', 'j').replace('jj', 'j')
    parts = [p.strip() for p in s.split(',') if p.strip()]
    if not parts:
        raise ValueError("Input statevector string is empty.")

    vals = []
    for p in parts:
        try:
            val = complex(eval(p, {"np": np, "math": math}))
        except Exception:
            try:
                val = complex(p)
            except Exception:
                raise ValueError(f"Could not parse '{p}' as a complex number.")
        vals.append(val)

    sv = np.array(vals, dtype=np.complex128)
    if len(sv) == 2:
        det_n = 1
    elif len(sv) == 4:
        det_n = 2
    else:
        raise ValueError(f"Statevector must have length 2 (1Q) or 4 (2Q). Got {len(sv)}.")

    if n_qubits is not None and n_qubits != det_n:
        raise ValueError(f"--qubits {n_qubits} does not match input vector length {len(sv)}.")

    norm = np.linalg.norm(sv)
    if norm < 1e-12:
        raise ValueError("Statevector norm is zero — invalid input.")
    return sv / norm, det_n


# ─────────────────────────────────────────────────────────────────────────────
# Model Loading
# ─────────────────────────────────────────────────────────────────────────────

def find_dqn_model_path(n_qubits: int) -> str:
    """Find the best available DQN checkpoint for 1Q or 2Q."""
    if n_qubits == 1:
        candidates = [
            os.path.join(SCRIPT_DIR, 'quantumrl', 'saved_models', '1qubit', 'dqn_model_best.pth'),
            os.path.join(SCRIPT_DIR, 'quantumrl', 'saved_models', '1qubit', 'dqn_model.pth'),
            os.path.join(SCRIPT_DIR, 'saved_models', 'dqn_model_best.pth'),
        ]
    else:
        candidates = [
            os.path.join(SCRIPT_DIR, 'saved_models', '2qubit', 'dqn_model_best.pth'),
            os.path.join(SCRIPT_DIR, 'saved_models', '2qubit', 'dqn_model.pth'),
            os.path.join(SCRIPT_DIR, 'quantumrl', 'saved_models', '2qubit', 'dqn_model_best.pth'),
        ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"No DQN model checkpoint found for {n_qubits}Q. Searched: {candidates}"
    )


def load_dqn_agent(n_qubits: int, device: torch.device) -> Tuple[DQNAgent, QuantumCircuitEnv, object, str]:
    """Instantiate QuantumCircuitEnv and load DQNAgent weights for 1Q or 2Q."""
    config = Config1Qubit() if n_qubits == 1 else Config2Qubit()
    model_path = find_dqn_model_path(n_qubits)

    ckpt = torch.load(model_path, map_location='cpu', weights_only=False)

    # Extract state dict from various checkpoint formats
    if isinstance(ckpt, dict) and 'q_net' in ckpt:
        sd = ckpt['q_net']
    elif isinstance(ckpt, dict) and 'state_dict' in ckpt:
        sd = ckpt['state_dict']
    elif isinstance(ckpt, dict) and 'q_net_state_dict' in ckpt:
        sd = ckpt['q_net_state_dict']
    else:
        sd = ckpt  # assume direct state_dict

    env = QuantumCircuitEnv(config)
    obs_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = DQNAgent(obs_size=obs_size, action_size=action_size, config=config)
    agent.q_net.load_state_dict(sd, strict=False)
    agent.q_net.to(device)
    agent.q_net.eval()
    agent.device = device

    return agent, env, config, model_path


# ─────────────────────────────────────────────────────────────────────────────
# DQN Synthesis Engine
# ─────────────────────────────────────────────────────────────────────────────

def select_action_stochastic(q_values: np.ndarray, temperature: float = 1.0) -> int:
    """Boltzmann (softmax) sampling over Q-values for stochastic exploration."""
    q_scaled = q_values / max(temperature, 1e-3)
    exp_q = np.exp(q_scaled - np.max(q_scaled))
    probs = exp_q / np.sum(exp_q)
    return int(np.random.choice(len(probs), p=probs))


def run_dqn_synthesis(
    target_sv: np.ndarray,
    n_qubits: int,
    budget: int = 30,
    target_fidelity: Optional[float] = None,
):
    """
    Full DQN synthesis engine with multi-candidate search, structural shortening,
    L-BFGS-B continuous refinement, and fresh Qiskit verification.
    Mirrors the PPO synthesis pipeline exactly.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if target_fidelity is None:
        target_fidelity = 0.999999 if n_qubits == 2 else 0.999

    print()
    print("=" * 74)
    print(f"          TATVA DQN QUANTUM CIRCUIT SYNTHESIZER ({n_qubits}-QUBIT)")
    print("=" * 74)
    print(f" -> Device           : {device}")
    print(f" -> System           : {n_qubits}-Qubit(s)")

    agent, env, config, model_path = load_dqn_agent(n_qubits, device)
    rel_model_path = os.path.relpath(model_path, SCRIPT_DIR)
    print(f" -> Loaded DQN Model : {rel_model_path}")

    sv_display = ', '.join(
        f'{a.real:+.6f}{a.imag:+.6f}i'
        for a in target_sv
    )
    print(f" -> Target Vector    : [{sv_display}]")
    print(f" -> Target Fidelity  : {target_fidelity:.6f}")
    print(f" -> Candidate Budget : {budget}")
    print("-" * 74)

    # Normalize
    orig_norm = float(np.linalg.norm(target_sv))
    target_sv = target_sv / orig_norm if orig_norm > 1e-10 else target_sv

    print()
    print("=" * 66)
    print(f"TATVA DQN SYNTHESIS ENGINE (Target: F >= {target_fidelity})")
    print(f"Norm: {orig_norm:.16f} | Budget: {budget} candidates")
    print("=" * 66)

    t0 = time.time()
    candidate_pool = []

    for attempt in range(budget):
        deterministic = (attempt == 0)
        obs, _ = env.reset(target_sv=target_sv)
        done = False

        while not done:
            st = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                q_vals = agent.q_net(st).squeeze(0).cpu().numpy()

            if deterministic:
                action = int(np.argmax(q_vals))
            elif attempt % 2 == 1:
                eps = 0.05 + (attempt / budget) * 0.25
                if np.random.rand() < eps:
                    action = int(np.random.randint(len(q_vals)))
                else:
                    action = int(np.argmax(q_vals))
            else:
                temp = 0.3 + (attempt / budget) * 1.5
                action = select_action_stochastic(q_vals, temperature=temp)

            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

        raw_actions = list(env.applied_actions)
        raw_gate_count = len(raw_actions)

        # Step 1 — Continuous parameter refinement + simplification
        ver_res = verify_synthesis_candidate(
            target_sv=target_sv,
            applied_actions=raw_actions,
            n_qubits=env.n_qubits,
            fidelity_threshold=target_fidelity,
        )
        simp_actions = ver_res.get('simplified_actions', raw_actions)

        # Step 2 — Structural shortening: prune + L-BFGS-B re-optimize
        opt_actions, opt_fid = prune_and_reoptimize_structural_search(
            simp_actions,
            target_sv=target_sv,
            n_qubits=env.n_qubits,
            target_fidelity=target_fidelity,
        )

        # Step 3 — Fresh Qiskit verification + 4-way consistency check
        opt_f, int_f, fresh_qiskit_f, disp_f, max_dev = verify_4way_consistency(
            opt_actions,
            target_sv=target_sv,
            n_qubits=env.n_qubits,
            internal_fid=ver_res.get('fidelity'),
        )

        fresh_qc = build_qiskit_circuit(opt_actions, n_qubits=env.n_qubits)
        mode_str = "Greedy (Deterministic)" if deterministic else "Stochastic Exploration"

        print(f"  Attempt {attempt+1:02d}/{budget:02d} [{mode_str:24s}] -> "
              f"Gates: {raw_gate_count} -> {len(opt_actions)} | "
              f"Qiskit F = {fresh_qiskit_f:.10f} | Dev: {max_dev:.1e}")

        status = (
            'HIGH_PRECISION_SUCCESS' if fresh_qiskit_f >= target_fidelity
            else ('APPROXIMATE_SUCCESS' if fresh_qiskit_f >= 0.99
            else 'FAILED')
        )

        cand_rec = {
            'attempt': attempt + 1,
            'deterministic': deterministic,
            'raw_actions': raw_actions,
            'raw_gate_count': raw_gate_count,
            'ver_res': ver_res,
            'final_fidelity': fresh_qiskit_f,
            'optimizer_fidelity': opt_f,
            'internal_fidelity': int_f,
            'displayed_circuit_fidelity': disp_f,
            'max_discrepancy': max_dev,
            'status': status,
            'verified': (fresh_qiskit_f >= target_fidelity and max_dev < 1e-10),
            'simplified_actions': opt_actions,
            'final_gate_count': len(opt_actions),
            'circuit_depth': fresh_qc.depth(),
            'entangling_gate_count': sum(1 for g, q, a in opt_actions if g in ('CZ', 'CNOT', 'SWAP')),
            'full_precision_parameters': [(g, q, float(a) if a is not None else None) for g, q, a in opt_actions],
            'display_lines': format_circuit_display_lines(opt_actions),
            'search_time_seconds': time.time() - t0,
            'qiskit_circuit': fresh_qc,
        }
        candidate_pool.append(cand_rec)

        # Early exit on minimal high-precision solution
        if fresh_qiskit_f >= target_fidelity and cand_rec['final_gate_count'] <= 12:
            print(f"  [EARLY STOP] High-precision minimal circuit found "
                  f"({cand_rec['final_gate_count']} gates, F={fresh_qiskit_f:.12f})")
            break

    # Rank: fidelity first, then gate count, then depth
    candidate_pool.sort(key=lambda c: candidate_rank_key(c, target_fidelity=target_fidelity))
    best = candidate_pool[0]

    # Final 4-way consistency assertion on winner
    opt_f, int_f, final_f, disp_f, max_dev = verify_4way_consistency(
        best['simplified_actions'],
        target_sv=target_sv,
        n_qubits=env.n_qubits,
        internal_fid=best['internal_fidelity'],
    )
    best['candidate_pool_size'] = len(candidate_pool)
    best['search_time_seconds'] = time.time() - t0

    # ── Results ──────────────────────────────────────────────────────────────
    gate_reduction = best['raw_gate_count'] - best['final_gate_count']
    reduction_pct = gate_reduction / best['raw_gate_count'] * 100 if best['raw_gate_count'] > 0 else 0

    print()
    print("=" * 74)
    print("      TATVA OPTIMIZED SYNTHESIS RESULTS (DQN + STRUCTURAL SEARCH)")
    print("=" * 74)
    print(f" Status                 : {best['status']}")
    print(f" Final Verified Fidelity: {best['final_fidelity']:.10f} ({best['final_fidelity']*100:.8f}%)")
    print(f" Raw DQN Gate Count     : {best['raw_gate_count']}")
    print(f" Final Optimized Gates  : {best['final_gate_count']}  (Reduced by {gate_reduction} gates / {reduction_pct:.2f}%)")
    print(f" Circuit Depth          : {best['circuit_depth']}")
    print(f" Entangling Gates       : {best['entangling_gate_count']}")
    print(f" 4-Way Consistency Dev  : {max_dev:.2e}")
    print(f" Candidates Evaluated   : {best['candidate_pool_size']}")
    print(f" Search & Opt Time      : {best['search_time_seconds']:.2f} seconds")
    print("-" * 74)
    if best['display_lines']:
        print(" Synthesized Gate Sequence:")
        for line in best['display_lines']:
            print(line.replace('\u03b8', 'theta'))
    else:
        print(" Gate Sequence          : [Identity / Empty Circuit]")
    print("=" * 74)
    print()

    return best


# ─────────────────────────────────────────────────────────────────────────────
# Main CLI Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="TATVA DQN Quantum Circuit Synthesizer (1-Qubit & 2-Qubit)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python dqn_synthesize.py --qubits 2 --target bell
  python dqn_synthesize.py --qubits 1 --target plus
  python dqn_synthesize.py --qubits 2 --target "0.5, 0.5, 0.5, 0.5"
  python dqn_synthesize.py --qubits 2 --target "-0.7989-0.4544i, 0.0354-0.2684i, -0.0138+0.1217i, -0.1021+0.2377i"
  python dqn_synthesize.py --qubits 1 --target random --budget 10
        """
    )
    parser.add_argument('-q', '--qubits', type=int, choices=[1, 2], default=None,
                        help='Number of qubits (1 or 2). Auto-detected from input if not given.')
    parser.add_argument('-t', '--target', type=str, default=None,
                        help="Target statevector: preset name ('bell', 'plus' ...) or comma-separated amplitudes.")
    parser.add_argument('-b', '--budget', type=int, default=30,
                        help='Number of DQN candidate rollouts (default: 30).')
    parser.add_argument('-f', '--fidelity-threshold', type=float, default=None,
                        help='Target fidelity (default: 0.999 for 1Q, 0.999999 for 2Q).')

    args = parser.parse_args()

    # CLI mode vs Interactive mode
    if args.target is not None:
        target_str = args.target
        n_qubits = args.qubits
    else:
        print()
        print("=" * 70)
        print("    WELCOME TO TATVA DQN QUANTUM CIRCUIT SYNTHESIZER")
        print("=" * 70)

        if args.qubits is None:
            print(" Select System:")
            print("   1) 1-Qubit System")
            print("   2) 2-Qubit System")
            q_choice = input(" Choice [1 or 2, default: 2]: ").strip()
            n_qubits = 1 if q_choice == '1' else 2
        else:
            n_qubits = args.qubits

        print(f"\n Enter Target Statevector for {n_qubits}-Qubit System.")
        if n_qubits == 1:
            print(" Presets  : 'plus', 'minus', 'zero', 'one', 'i_state', 'random'")
            print(" Custom   : '0.70710678, 0.70710678'  or  '1/sqrt(2), 1i/sqrt(2)'")
        else:
            print(" Presets  : 'bell', 'phi_plus', 'phi_minus', 'psi_plus', 'psi_minus', 'zero', 'plus', 'random'")
            print(" Custom   : '0.5, 0.5, 0.5, 0.5'  or  '0.70710678, 0, 0, 0.70710678'")

        target_str = input("\n Target Statevector: ").strip()
        if not target_str:
            target_str = 'bell' if n_qubits == 2 else 'plus'
            print(f" [No input -> using default preset '{target_str}']")

    try:
        target_sv, n_qubits = parse_statevector(target_str, n_qubits=n_qubits)
    except Exception as e:
        print(f"\n [ERROR] Failed to parse target statevector: {e}")
        sys.exit(1)

    run_dqn_synthesis(
        target_sv=target_sv,
        n_qubits=n_qubits,
        budget=args.budget,
        target_fidelity=args.fidelity_threshold,
    )


if __name__ == '__main__':
    main()
