# week3_probe.py

import os
import sys
import numpy as np

# Fix import path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from week2.week2_reinforce import train_reinforce


def run_experiment(name, learning_rate, gamma):
    print(f"\n===== {name} =====")
    print(f"learning_rate={learning_rate}, gamma={gamma}")

    returns, _ = train_reinforce(
        learning_rate=learning_rate,
        gamma=gamma,
        num_episodes=300,
    )

    last_10_avg = np.mean(returns[-10:])
    print(f"Final 10-episode average return: {last_10_avg:.2f}")


if __name__ == "__main__":

    # Baseline (Week 2)
    run_experiment(
        name="Baseline",
        learning_rate=1e-2,
        gamma=0.99,
    )

    # Run A: Too large learning rate
    run_experiment(
        name="Run A (lr=0.1)",
        learning_rate=0.1,
        gamma=0.99,
    )

    # Run B: Too small learning rate
    run_experiment(
        name="Run B (lr=1e-5)",
        learning_rate=1e-5,
        gamma=0.99,
    )

    # Run C: Low gamma (short-sighted)
    run_experiment(
        name="Run C (gamma=0.5)",
        learning_rate=1e-2,
        gamma=0.5,
    )