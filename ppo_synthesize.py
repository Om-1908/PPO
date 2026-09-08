"""
ppo_synthesize.py
-----------------
Standalone Terminal Inference Script for TATVA PPO Quantum Circuit Synthesis.

Supports both 1-Qubit and 2-Qubit systems using trained PPO agents ONLY.

Usage:
------
  1. Interactive mode (run directly in terminal):
         python ppo_synthesize.py

  2. Command line interface (CLI mode):
         python ppo_synthesize.py --qubits 1 --target plus
         python ppo_synthesize.py --qubits 2 --target bell
         python ppo_synthesize.py --qubits 2 --target "0.5, 0.5, 0.5, 0.5"
         python ppo_synthesize.py --qubits 1 --target "1/sqrt(2), 1i/sqrt(2)"
"""

import argparse
import math
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

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
from ppo_agent import PPOAgent
from utils import generate_random_statevector
from synthesis import synthesize_circuit, build_qiskit_circuit, format_circuit_display_lines

# ─────────────────────────────────────────────────────────────────────────────
# Named Presets
# ─────────────────────────────────────────────────────────────────────────────

INV_SQRT2 = 1.0 / math.sqrt(2.0)

PRESETS_1Q = {
    'zero':    np.array([1.0+0j, 0.0+0j],         dtype=np.complex128),
    'one':     np.array([0.0+0j, 1.0+0j],         dtype=np.complex128),
    'plus':    np.array([INV_SQRT2, INV_SQRT2],   dtype=np.complex128),
    'minus':   np.array([INV_SQRT2, -INV_SQRT2],  dtype=np.complex128),
    'i_state': np.array([INV_SQRT2, INV_SQRT2*1j], dtype=np.complex128),
}

PRESETS_2Q = {
    'bell':      np.array([INV_SQRT2, 0.0+0j, 0.0+0j, INV_SQRT2],   dtype=np.complex128),
    'phi_plus':  np.array([INV_SQRT2, 0.0+0j, 0.0+0j, INV_SQRT2],   dtype=np.complex128),
    'phi_minus': np.array([INV_SQRT2, 0.0+0j, 0.0+0j, -INV_SQRT2],  dtype=np.complex128),
    'psi_plus':  np.array([0.0+0j, INV_SQRT2, INV_SQRT2, 0.0+0j],   dtype=np.complex128),
    'psi_minus': np.array([0.0+0j, INV_SQRT2, -INV_SQRT2, 0.0+0j],  dtype=np.complex128),
    'ghz':       np.array([INV_SQRT2, 0.0+0j, 0.0+0j, INV_SQRT2],   dtype=np.complex128),
    'zero':      np.array([1.0+0j, 0.0+0j, 0.0+0j, 0.0+0j],         dtype=np.complex128),
    'plus':      np.array([0.5+0j, 0.5+0j, 0.5+0j, 0.5+0j],          dtype=np.complex128),
}


def parse_statevector(input_str: str, n_qubits: Optional[int] = None) -> Tuple[np.ndarray, int]:
    """
    Parse a user input string into a normalized complex128 statevector.
    Supports preset names ('bell', 'plus', etc.), 'random', or custom amplitude lists.
    """
    clean_str = input_str.strip().lower()

    if clean_str == 'random':
        target_n = n_qubits if n_qubits in (1, 2) else 2
        sv = generate_random_statevector(target_n)
        return sv, target_n

    # Check 1-qubit presets
    if n_qubits == 1 and clean_str in PRESETS_1Q:
        return PRESETS_1Q[clean_str], 1
    if n_qubits == 2 and clean_str in PRESETS_2Q:
        return PRESETS_2Q[clean_str], 2
    if n_qubits is None:
        if clean_str in PRESETS_1Q:
            return PRESETS_1Q[clean_str], 1
        if clean_str in PRESETS_2Q:
            return PRESETS_2Q[clean_str], 2

    # Custom vector parsing
    s = clean_str.replace('[', '').replace(']', '').replace('(', '').replace(')', '')
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
                raise ValueError(f"Could not parse element '{p}' as a complex number.")
        vals.append(val)

    sv = np.array(vals, dtype=np.complex128)
    length = len(sv)

    if length == 2:
        detected_n = 1
    elif length == 4:
        detected_n = 2
    else:
        raise ValueError(f"Statevector length must be 2 (1-qubit) or 4 (2-qubit). Got length {length}.")

    if n_qubits is not None and n_qubits != detected_n:
        raise ValueError(f"Specified --qubits {n_qubits} does not match input vector length {length}.")

    norm = np.linalg.norm(sv)
    if norm < 1e-12:
        raise ValueError("Statevector norm cannot be zero.")

    normalized_sv = sv / norm
    return normalized_sv, detected_n


def find_ppo_model_path(n_qubits: int) -> str:
    """Find the best available PPO checkpoint for 1-qubit or 2-qubit system."""
    if n_qubits == 1:
        candidates = [
            os.path.join(SCRIPT_DIR, 'saved_models', 'ppo_model.pth'),
            os.path.join(SCRIPT_DIR, 'quantumrl', 'saved_models', '1qubit', 'ppo_model.pth'),
            os.path.join(SCRIPT_DIR, 'saved_models', 'ppo_model_best.pth'),
        ]
    else:
        candidates = [
            os.path.join(SCRIPT_DIR, 'checkpoints', 'ppo_paused_ep174500.pth'),
            os.path.join(SCRIPT_DIR, 'saved_models', '2qubit', 'ppo_model_best_best.pth'),
            os.path.join(SCRIPT_DIR, 'saved_models', '2qubit', 'ppo_model_best.pth'),
            os.path.join(SCRIPT_DIR, 'saved_models', '2qubit', 'ppo_pretrained_best.pth'),
        ]

    for p in candidates:
        if os.path.exists(p):
            return p

    raise FileNotFoundError(f"No PPO model checkpoint found for {n_qubits}-qubit system in candidates: {candidates}")


def load_ppo_agent(n_qubits: int, device: torch.device) -> Tuple[PPOAgent, QuantumCircuitEnv, object, str]:
    """Instantiate QuantumCircuitEnv and load PPOAgent weights for 1Q or 2Q."""
    if n_qubits == 1:
        config = Config1Qubit()
    else:
        config = Config2Qubit()

    model_path = find_ppo_model_path(n_qubits)

    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "actor_critic_state_dict" in ckpt:
        sd = ckpt["actor_critic_state_dict"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    else:
        sd = ckpt

    # Adapt legacy base.* keys if present
    if any(k.startswith('base.') for k in sd):
        adapted_sd = {}
        for k, v in sd.items():
            if k.startswith('base.'):
                sub_key = k[len('base.'):]
                adapted_sd[f'actor.{sub_key}'] = v
                adapted_sd[f'critic.{sub_key}'] = v
            elif k.startswith('actor_head.'):
                sub_key = k[len('actor_head.'):]
                adapted_sd[f'actor.9.{sub_key}'] = v
            elif k.startswith('critic_head.'):
                sub_key = k[len('critic_head.'):]
                adapted_sd[f'critic.9.{sub_key}'] = v
            else:
                adapted_sd[k] = v
        sd = adapted_sd

    # Detect hidden layer dimension from state dict
    if 'actor.0.weight' in sd:
        h1 = sd['actor.0.weight'].shape[0]
        config.PPO_HIDDEN_SIZE = h1

    # Detect action space size from state dict
    if 'actor.9.weight' in sd:
        act_size = sd['actor.9.weight'].shape[0]
        if n_qubits == 1 and act_size == 76:
            config.ROTATION_ANGLES = [k * math.pi / 12 for k in range(1, 25)]
        elif n_qubits == 2 and act_size == 164:
            config.ROTATION_ANGLES = [
                k * math.pi / 32 for k in range(1, 65)
            ] + [-k * math.pi / 32 for k in range(1, 33)]

    env = QuantumCircuitEnv(config)
    obs_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = PPOAgent(obs_size=obs_size, action_size=action_size, config=config, device=device)
    agent.ac.load_state_dict(sd, strict=False)
    agent.ac.eval()
    return agent, env, config, model_path


def format_statevector_str(sv: np.ndarray) -> str:
    """Format statevector array cleanly for terminal display."""
    formatted = []
    for idx, amp in enumerate(sv):
        real_part = f"{amp.real:+.6f}"
        imag_part = f"{amp.imag:+.6f}i"
        formatted.append(f"|{idx:0{len(bin(len(sv)-1))-2}b}>: {real_part} {imag_part}")
    return " ,  ".join(formatted)


def run_ppo_synthesis(
    target_sv: np.ndarray,
    n_qubits: int,
    budget: int = 30,
    target_fidelity: Optional[float] = None
):
    """Run PPO synthesis engine and display results."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n" + "=" * 74)
    print(f"          TATVA PPO QUANTUM CIRCUIT SYNTHESIZER ({n_qubits}-QUBIT)")
    print("=" * 74)

    print(f" -> Device           : {device}")
    print(f" -> System           : {n_qubits}-Qubit(s)")
    
    agent, env, config, model_path = load_ppo_agent(n_qubits, device)
    rel_model_path = os.path.relpath(model_path, SCRIPT_DIR)
    print(f" -> Loaded PPO Model : {rel_model_path}")
    print(f" -> Target Vector    : {target_sv.tolist()}")
    print(f" -> Amplitudes       : {format_statevector_str(target_sv)}")

    if target_fidelity is None:
        target_fidelity = 0.999999 if n_qubits == 2 else 0.999

    print(f" -> Target Fidelity  : {target_fidelity:.6f}")
    print(f" -> Candidate Budget : {budget}")
    print("-" * 74)
    print(f" Running PPO synthesis search ...")

    t_start = time.time()
    result = synthesize_circuit(
        target_sv=target_sv,
        agent=agent,
        env=env,
        config=config,
        target_fidelity=target_fidelity,
        max_candidates=budget,
        verbose=True
    )
    elapsed = time.time() - t_start

    final_fid = result['final_fidelity']
    display_lines = result['display_lines']
    gate_count = result['final_gate_count']
    depth = result['circuit_depth']
    status = result['status']

    print("\n" + "=" * 74)
    print("                      SYNTHESIS RESULTS (PPO ONLY)")
    print("=" * 74)
    print(f" Status            : {status}")
    print(f" Final Fidelity    : {final_fid:.10f} ({final_fid * 100:.6f}%)")
    print(f" Total Gate Count  : {gate_count}")
    print(f" Circuit Depth     : {depth}")
    print(f" Search Time       : {elapsed:.2f} seconds")
    print("-" * 74)

    if display_lines:
        print(" Synthesized Gate Sequence:")
        for line in display_lines:
            print(line.replace('θ', 'theta'))
    else:
        print(" Gate Sequence     : [Identity / Empty Circuit]")

    print("=" * 74 + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(
        description="TATVA PPO Quantum Circuit Synthesizer (1-Qubit & 2-Qubit)"
    )
    parser.add_argument(
        "-q", "--qubits", type=int, choices=[1, 2], default=None,
        help="Number of qubits (1 or 2). Auto-detected if not specified."
    )
    parser.add_argument(
        "-t", "--target", type=str, default=None,
        help="Target statevector: preset name ('bell', 'plus', etc.) or comma-separated amplitudes."
    )
    parser.add_argument(
        "-b", "--budget", type=int, default=30,
        help="Number of PPO policy candidate trajectories to evaluate (default: 30)."
    )
    parser.add_argument(
        "-f", "--fidelity-threshold", type=float, default=None,
        help="Target fidelity threshold (default: 0.999 for 1Q, 0.999999 for 2Q)."
    )

    args = parser.parse_args()

    # CLI mode vs Interactive Terminal mode
    if args.target is not None:
        target_str = args.target
        n_qubits = args.qubits
    else:
        print("\n" + "=" * 70)
        print("    WELCOME TO TATVA PPO QUANTUM CIRCUIT SYNTHESIZER")
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
            print(" Available presets : 'plus', 'minus', 'zero', 'one', 'i_state', 'random'")
            print(" Amplitude example : '0.70710678, 0.70710678' or '1/sqrt(2), 1i/sqrt(2)'")
        else:
            print(" Available presets : 'bell', 'phi_plus', 'phi_minus', 'psi_plus', 'psi_minus', 'zero', 'plus', 'random'")
            print(" Amplitude example : '0.5, 0.5, 0.5, 0.5' or '0.70710678, 0, 0, 0.70710678'")

        target_str = input("\n Input Target Statevector: ").strip()
        if not target_str:
            target_str = 'bell' if n_qubits == 2 else 'plus'
            print(f" [No input provided -> using default preset '{target_str}']")

    try:
        target_sv, n_qubits = parse_statevector(target_str, n_qubits=n_qubits)
    except Exception as err:
        print(f"\n [ERROR] Failed to parse target statevector: {err}")
        sys.exit(1)

    run_ppo_synthesis(
        target_sv=target_sv,
        n_qubits=n_qubits,
        budget=args.budget,
        target_fidelity=args.fidelity_threshold
    )


if __name__ == '__main__':
    main()
