# week3_compare.py

import numpy as np
import matplotlib.pyplot as plt
import os


def moving_average(x, window=10):
    x = np.array(x)
    if len(x) < window:
        return x
    return np.convolve(x, np.ones(window) / window, mode="valid")


def plot_comparison(reinforce_returns, ppo_returns, window=10, filename="plots/reinforce_vs_ppo.png"):
    os.makedirs("plots", exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 4))

    # REINFORCE
    r = np.array(reinforce_returns)
    ax.plot(r, alpha=0.2)
    ax.plot(moving_average(r, window), label=f"REINFORCE ({window}-ep avg)")

    # PPO
    p = np.array(ppo_returns)
    ax.plot(p, alpha=0.2)
    ax.plot(moving_average(p, window), label=f"PPO ({window}-ep avg)")

    ax.set_xlabel("Episode")
    ax.set_ylabel("Return")
    ax.set_title("REINFORCE vs PPO on CartPole-v1")
    ax.legend()

    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)
    print(f"Saved to {filename}")


if __name__ == "__main__":
    reinforce_returns = np.load("reinforce_returns.npy").tolist()
    ppo_returns = np.load("ppo_returns.npy").tolist()
    plot_comparison(reinforce_returns, ppo_returns)