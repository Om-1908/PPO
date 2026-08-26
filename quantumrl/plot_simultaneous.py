"""
plot_simultaneous.py
--------------------
Utility script to generate simultaneous training comparison graphs for DQN and PPO.
Reads logs/2qubit/dqn_logs.json and logs/2qubit/ppo_logs.json and saves the plot to plots/2qubit/simultaneous_dqn_ppo.png.

Run with:
    python plot_simultaneous.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from utils import plot_simultaneous_curves


def main():
    cfg = Config()
    dqn_log = os.path.join(cfg.LOG_DIR, 'dqn_logs.json')
    ppo_log = getattr(cfg, 'PPO_LOG_PATH', os.path.join(cfg.LOG_DIR, 'ppo_logs.json'))
    sim_plot_path = os.path.join(cfg.PLOT_DIR, 'simultaneous_dqn_ppo.png')

    print(f"[plot_simultaneous] Generating simultaneous plot...")
    print(f"  DQN Log Path : {dqn_log}")
    print(f"  PPO Log Path : {ppo_log}")
    print(f"  Plot Target  : {sim_plot_path}")

    plot_simultaneous_curves(dqn_log, ppo_log, sim_plot_path)


if __name__ == '__main__':
    main()
