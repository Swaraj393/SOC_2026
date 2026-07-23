import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from week6_historical_env import HistoricalTradingEnv


# ============================================================
# Helper Functions
# ============================================================

def run_episode(env, model=None, strategy="trained"):
    """
    Runs one episode and returns the reward at each step.
    """

    obs, _ = env.reset()
    rewards = []

    done = False

    while not done:

        if strategy == "trained":
            action, _ = model.predict(obs, deterministic=True)

        elif strategy == "random":
            action = env.action_space.sample()

        elif strategy == "buy_and_hold":
            action = 0

        else:
            raise ValueError("Unknown strategy")

        obs, reward, terminated, truncated, _ = env.step(action)

        rewards.append(reward)

        done = terminated or truncated

    return rewards


def average_cumulative_pnl(
    env,
    n_episodes=50,
    model=None,
    strategy="trained",
):
    """
    Mean cumulative P&L over multiple episodes.
    """

    curves = []

    for _ in range(n_episodes):

        rewards = run_episode(
            env,
            model=model,
            strategy=strategy,
        )

        curves.append(np.cumsum(rewards))

    min_len = min(len(c) for c in curves)

    curves = np.array([c[:min_len] for c in curves])

    return curves.mean(axis=0)


def plot_results(
    trained,
    random_curve,
    bnh,
    title,
    filename,
):

    os.makedirs("plots", exist_ok=True)

    plt.figure(figsize=(10, 5))

    plt.plot(trained, label="PPO")

    plt.plot(random_curve, label="Random")

    plt.plot(bnh, label="Buy & Hold")

    plt.axhline(0, linestyle="--")

    plt.xlabel("Step")

    plt.ylabel("Cumulative P&L")

    plt.title(title)

    plt.legend()

    plt.tight_layout()

    plt.savefig(filename)

    plt.close()

    print(f"Saved plot -> {filename}")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    BASE_DIR = Path(__file__).parent

    DATA_DIR = BASE_DIR / "data"

    MODELS_DIR = BASE_DIR / "models"

    PLOTS_DIR = BASE_DIR / "plots"

    os.makedirs(PLOTS_DIR, exist_ok=True)

    print("=" * 60)
    print("Evaluating all trained models")
    print("=" * 60)

    model_files = sorted(MODELS_DIR.glob("*.zip"))

    if len(model_files) == 0:
        raise FileNotFoundError(
            "No trained models found inside models/ folder."
        )

    all_scores = []

    for model_path in model_files:

        name = model_path.stem

        print("\n" + "-" * 60)
        print(name)
        print("-" * 60)

        # ---------------------------------------------------

        if name.startswith("spy"):

            returns_path = DATA_DIR / "spy_returns.npy"

        elif name.startswith("nifty"):

            returns_path = DATA_DIR / "nifty_returns.npy"

        else:

            print("Unknown model name. Skipping.")

            continue

        returns = np.load(returns_path)

        split = int(0.8 * len(returns))

        test_returns = returns[split:]

        env = HistoricalTradingEnv(test_returns)

        model = PPO.load(model_path)

        pnl_trained = average_cumulative_pnl(
            env,
            model=model,
            strategy="trained",
        )

        pnl_random = average_cumulative_pnl(
            env,
            strategy="random",
        )

        pnl_bnh = average_cumulative_pnl(
            env,
            strategy="buy_and_hold",
        )

        trained_score = pnl_trained[-1]

        random_score = pnl_random[-1]

        bnh_score = pnl_bnh[-1]

        print(f"PPO         : {trained_score:8.2f}")
        print(f"Random      : {random_score:8.2f}")
        print(f"Buy & Hold  : {bnh_score:8.2f}")

        all_scores.append(trained_score)

        plot_file = PLOTS_DIR / f"{name}_evaluation.png"

        plot_results(
            pnl_trained,
            pnl_random,
            pnl_bnh,
            title=name,
            filename=plot_file,
        )

    print("\n" + "=" * 60)

    print("Summary Across All Models")

    print("=" * 60)

    print(f"Number of models : {len(all_scores)}")

    print(f"Mean PPO P&L     : {np.mean(all_scores):.2f}")

    print(f"Std PPO P&L      : {np.std(all_scores):.2f}")

    print("=" * 60)