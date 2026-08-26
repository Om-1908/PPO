"""
train_both.py
-------------
Orchestrator script to train both DQN and PPO for 2 qubits (50,000 episodes each)
and generate simultaneous diagnostic plots.

Run with:
    python train_both.py
"""

import os
import sys
import multiprocessing
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from train_dqn import train_dqn
from train_ppo import train_ppo
from plot_simultaneous import main as plot_simultaneous_main


def run_dqn():
    cfg = Config()
    print("[train_both] Starting DQN Training Process...")
    train_dqn(cfg)


def run_ppo():
    cfg = Config()
    print("[train_both] Starting PPO Training Process...")
    train_ppo(cfg)


if __name__ == '__main__':
    os.environ['PYTHONUNBUFFERED'] = '1'
    cfg = Config()
    print(f"============================================================", flush=True)
    print(f" QuantumRL: Starting 2-Qubit {cfg.DQN_EPISODES}-Episode Dual Training ", flush=True)
    print(f"============================================================", flush=True)

    p_dqn = multiprocessing.Process(target=run_dqn)
    p_ppo = multiprocessing.Process(target=run_ppo)

    p_dqn.start()
    p_ppo.start()

    print("[train_both] Processes spawned for both DQN and PPO.", flush=True)

    # Periodic graph refresher process monitor
    try:
        while p_dqn.is_alive() or p_ppo.is_alive():
            time.sleep(30)
            try:
                plot_simultaneous_main()
            except Exception as e:
                pass
    except KeyboardInterrupt:
        print("[train_both] Interrupted by user.")

    p_dqn.join()
    p_ppo.join()

    print("[train_both] Final simultaneous plot update...")
    plot_simultaneous_main()
    print("============================================================")
    print(" Dual Training (DQN & PPO) Completed Successfully! ")
    print("============================================================")
