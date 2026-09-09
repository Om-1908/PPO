import os
import sys
import time
import math
import numpy as np
import torch
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

# Add quantumrl and scratch to path
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('quantumrl'))
sys.path.insert(0, os.path.abspath('scratch'))

from configs.config_2qubit import Config
from quantum_env import QuantumCircuitEnv
from dqn_agent import DQNAgent
from simplify import (
    verify_synthesis_candidate,
    simplify_gate_sequence,
    optimize_circuit_parameters,
    multi_start_optimize_parameters,
    _fidelity
)
from synthesis import (
    build_qiskit_circuit,
    verify_4way_consistency,
    prune_and_reoptimize_structural_search,
    candidate_rank_key
)
from scratch.test_20_references import TARGETS

def select_action_stochastic(q_values_np: np.ndarray, temperature: float = 1.0) -> int:
    """Softmax sampling over Q-values for stochastic policy search."""
    q_scaled = q_values_np / max(temperature, 1e-3)
    exp_q = np.exp(q_scaled - np.max(q_scaled))
    probs = exp_q / np.sum(exp_q)
    return int(np.random.choice(len(probs), p=probs))

def synthesize_target_dqn(
    target_sv: np.ndarray,
    agent: DQNAgent,
    env: QuantumCircuitEnv,
    config: Config,
    target_fidelity: float = 0.999999,
    max_candidates: int = 25,
    verbose: bool = True
) -> dict:
    t0 = time.time()
    target_norm = target_sv / np.linalg.norm(target_sv)
    candidate_pool = []

    for attempt in range(max_candidates):
        deterministic = (attempt == 0)
        obs, _ = env.reset(target_sv=target_norm)
        done = False

        while not done:
            state_t = torch.as_tensor(obs, dtype=torch.float32, device=agent.device).unsqueeze(0)
            with torch.no_grad():
                q_vals = agent.q_net(state_t).squeeze(0).cpu().numpy()

            if deterministic:
                action = int(np.argmax(q_vals))
            else:
                if attempt % 2 == 1:
                    # Epsilon greedy sampling
                    eps = 0.05 + (attempt / max_candidates) * 0.25
                    if np.random.rand() < eps:
                        action = int(np.random.randint(len(q_vals)))
                    else:
                        action = int(np.argmax(q_vals))
                else:
                    # Softmax temperature sampling
                    temp = 0.3 + (attempt / max_candidates) * 1.2
                    action = select_action_stochastic(q_vals, temperature=temp)

            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

        raw_actions = list(env.applied_actions)
        raw_gate_count = len(raw_actions)

        # Refine continuous parameters and simplify
        ver_res = verify_synthesis_candidate(
            target_sv=target_norm,
            applied_actions=raw_actions,
            n_qubits=env.n_qubits,
            fidelity_threshold=target_fidelity
        )
        simp_actions = ver_res.get('simplified_actions', raw_actions)

        # Multi-strategy structural shortening
        opt_actions, opt_fid = prune_and_reoptimize_structural_search(
            simp_actions,
            target_sv=target_norm,
            n_qubits=env.n_qubits,
            target_fidelity=target_fidelity
        )

        # 4-Way Consistency Verification
        opt_f, int_f, fresh_qiskit_f, disp_f, max_dev = verify_4way_consistency(
            opt_actions,
            target_sv=target_norm,
            n_qubits=env.n_qubits,
            internal_fid=ver_res.get('fidelity')
        )
        fresh_qc = build_qiskit_circuit(opt_actions, n_qubits=env.n_qubits)

        status = 'HIGH_PRECISION_SUCCESS' if fresh_qiskit_f >= target_fidelity else ('APPROXIMATE_SUCCESS' if fresh_qiskit_f >= 0.99 else 'FAILED')

        cand_rec = {
            'attempt': attempt + 1,
            'deterministic': deterministic,
            'raw_actions': raw_actions,
            'raw_gate_count': raw_gate_count,
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
            'search_time_seconds': time.time() - t0,
        }
        candidate_pool.append(cand_rec)

        if fresh_qiskit_f >= target_fidelity and cand_rec['final_gate_count'] <= 10:
            break

    candidate_pool.sort(key=lambda c: candidate_rank_key(c, target_fidelity=target_fidelity))
    best_cand = candidate_pool[0]
    best_cand['search_time_seconds'] = time.time() - t0
    best_cand['attempts'] = len(candidate_pool)
    return best_cand

def main():
    config = Config()
    env = QuantumCircuitEnv(config)
    agent = DQNAgent(env.observation_space.shape[0], env.action_space.n, config)
    
    ckpt_path = os.path.join('saved_models', '2qubit', 'dqn_model_best.pth')
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join('saved_models', '2qubit', 'dqn_model.pth')
    
    print(f"[DQN Evaluation] Loading DQN checkpoint from {ckpt_path}...", flush=True)
    agent.load(ckpt_path)
    agent.q_net.eval()

    results = []
    print("\n==========================================================================================================", flush=True)
    print("DQN 20-TARGET BENCHMARK EVALUATION", flush=True)
    print("==========================================================================================================", flush=True)
    print(f"{'Target':<8}{'Raw Gates':<11}{'Final Gates':<13}{'Depth':<8}{'Ent':<6}{'Qiskit Fid':<16}{'HP Success':<12}{'Ref Gates':<11}{'Time (s)':<10}", flush=True)
    print("----------------------------------------------------------------------------------------------------------", flush=True)

    hp_success_count = 0
    gate_counts = []
    depths = []
    entangling_counts = []
    fidelities = []

    for idx, (target_sv, ref_actions) in enumerate(TARGETS, 1):
        best_cand = synthesize_target_dqn(target_sv, agent, env, config, target_fidelity=0.999999, max_candidates=25, verbose=False)
        ref_gates = len(ref_actions)
        hp_success = best_cand['final_fidelity'] >= 0.999999
        if hp_success:
            hp_success_count += 1
            
        gate_counts.append(best_cand['final_gate_count'])
        depths.append(best_cand['circuit_depth'])
        entangling_counts.append(best_cand['entangling_gate_count'])
        fidelities.append(best_cand['final_fidelity'])

        print(
            f"T-{idx:02d}    "
            f"{best_cand['raw_gate_count']:<11}"
            f"{best_cand['final_gate_count']:<13}"
            f"{best_cand['circuit_depth']:<8}"
            f"{best_cand['entangling_gate_count']:<6}"
            f"{best_cand['final_fidelity']:<16.10f}"
            f"{str(hp_success):<12}"
            f"{ref_gates:<11}"
            f"{best_cand['search_time_seconds']:<10.2f}",
            flush=True
        )

        results.append({
            'target': idx,
            'raw_gates': best_cand['raw_gate_count'],
            'final_gates': best_cand['final_gate_count'],
            'depth': best_cand['circuit_depth'],
            'entangling': best_cand['entangling_gate_count'],
            'fidelity': best_cand['final_fidelity'],
            'hp_success': hp_success,
            'ref_gates': ref_gates,
            'time': best_cand['search_time_seconds']
        })

    print("----------------------------------------------------------------------------------------------------------", flush=True)
    print(f"SUMMARY STATISTICS FOR 20 VALIDATION TARGETS:", flush=True)
    print(f" High-Precision Success Rate (F >= 0.999999): {hp_success_count}/20 ({hp_success_count/20*100:.1f}%)", flush=True)
    print(f" Mean Qiskit Fidelity: {np.mean(fidelities):.10f}", flush=True)
    print(f" Median Qiskit Fidelity: {np.median(fidelities):.10f}", flush=True)
    print(f" Mean Final Gate Count: {np.mean(gate_counts):.2f}", flush=True)
    print(f" Median Final Gate Count: {np.median(gate_counts):.1f}", flush=True)
    print(f" Mean Circuit Depth: {np.mean(depths):.2f}", flush=True)
    print(f" Mean Entangling Gates: {np.mean(entangling_counts):.2f}", flush=True)
    print("==========================================================================================================", flush=True)

if __name__ == '__main__':
    main()
