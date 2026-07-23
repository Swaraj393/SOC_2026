from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

from baseline_env import HistoricalTradingEnv
from improved_env import ImprovedTradingEnv
from metrics import episode_summary, mean_curve


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
PLOTS_DIR = BASE_DIR / "plots"
RESULTS_DIR = BASE_DIR / "results"

PLOTS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

MARKETS = {
    "spy": DATA_DIR / "spy_returns.npy",
    "nifty": DATA_DIR / "nifty_returns.npy",
}

SEEDS = [0, 1, 2]
EVAL_EPISODES = 50
EVAL_SEEDS = list(range(EVAL_EPISODES))

BASELINE_ENV_KWARGS = dict(
    episode_length=100,
    history_len=5,
    transaction_cost=0.1,
    reward_scale=100.0,
)

IMPROVED_ENV_KWARGS = dict(
    episode_length=100,
    history_len=5,
    feature_window=10,
    transaction_cost=0.1,
    reward_scale=100.0,
)


def chronological_split(returns, train_fraction=0.8):
    split = int(len(returns) * train_fraction)
    return returns[:split], returns[split:]


def run_episode(env, strategy="trained", model=None, seed=None):
    if seed is not None:
        obs, _ = env.reset(seed=seed)
        env.action_space.seed(seed)
    else:
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
            raise ValueError(f"Unknown strategy: {strategy}")

        obs, reward, terminated, truncated, _ = env.step(int(action))
        rewards.append(float(reward))
        done = terminated or truncated

    return np.asarray(rewards, dtype=float)


def evaluate_strategy(env_cls, returns, env_kwargs, strategy, model=None):
    env = env_cls(returns, **env_kwargs)

    final_pnls = []
    sharpes = []
    drawdowns = []
    curves = []

    for seed in EVAL_SEEDS:
        rewards = run_episode(env, strategy=strategy, model=model, seed=seed)
        stats = episode_summary(rewards)

        final_pnls.append(stats["final_pnl"])
        sharpes.append(stats["sharpe"])
        drawdowns.append(stats["max_drawdown"])
        curves.append(stats["curve"])

    final_pnls = np.asarray(final_pnls, dtype=float)
    sharpes = np.asarray(sharpes, dtype=float)
    drawdowns = np.asarray(drawdowns, dtype=float)

    return {
        "final_pnl_mean": float(final_pnls.mean()),
        "final_pnl_std": float(final_pnls.std()),
        "sharpe_mean": float(sharpes.mean()),
        "sharpe_std": float(sharpes.std()),
        "max_drawdown_mean": float(drawdowns.mean()),
        "max_drawdown_std": float(drawdowns.std()),
        "curve": mean_curve(curves),
    }


def evaluate_seeded_models(env_cls, returns, env_kwargs, model_paths):
    per_seed_summaries = []
    curves = []

    if len(model_paths) == 0:
        raise FileNotFoundError("No model files found for this configuration.")

    for model_path in model_paths:
        model = PPO.load(str(model_path))
        result = evaluate_strategy(env_cls, returns, env_kwargs, "trained", model=model)
        per_seed_summaries.append(result)
        curves.append(result["curve"])

    summary = {
        "final_pnl_mean": float(np.mean([x["final_pnl_mean"] for x in per_seed_summaries])),
        "final_pnl_std": float(np.std([x["final_pnl_mean"] for x in per_seed_summaries])),
        "sharpe_mean": float(np.mean([x["sharpe_mean"] for x in per_seed_summaries])),
        "sharpe_std": float(np.std([x["sharpe_mean"] for x in per_seed_summaries])),
        "max_drawdown_mean": float(np.mean([x["max_drawdown_mean"] for x in per_seed_summaries])),
        "max_drawdown_std": float(np.std([x["max_drawdown_mean"] for x in per_seed_summaries])),
        "curve": mean_curve(curves),
    }
    return summary, per_seed_summaries


def save_four_curve_plot(curves, title, filename):
    plt.figure(figsize=(10, 5))
    for label, curve in curves.items():
        plt.plot(curve, label=label)
    plt.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    plt.xlabel("Step within episode")
    plt.ylabel("Mean cumulative P&L")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved plot -> {filename}")


def save_train_test_plot(train_curve, test_curve, title, filename):
    plt.figure(figsize=(10, 5))
    plt.plot(train_curve, label="Improved PPO (train)")
    plt.plot(test_curve, label="Improved PPO (test)")
    plt.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    plt.xlabel("Step within episode")
    plt.ylabel("Mean cumulative P&L")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved plot -> {filename}")


def print_row(name, summary):
    print(
        f"{name:18s} | "
        f"P&L {summary['final_pnl_mean']:+8.2f} ± {summary['final_pnl_std']:.2f} | "
        f"Sharpe {summary['sharpe_mean']:+6.2f} ± {summary['sharpe_std']:.2f} | "
        f"MaxDD {summary['max_drawdown_mean']:+7.2f} ± {summary['max_drawdown_std']:.2f}"
    )


def evaluate_market(market):
    returns = np.load(MARKETS[market])
    train_returns, test_returns = chronological_split(returns)

    baseline_paths = sorted(MODELS_DIR.glob(f"clean_{market}_baseline_seed_*.zip"))
    improved_paths = sorted(MODELS_DIR.glob(f"clean_{market}_improved_seed_*.zip"))

    print("\n" + "=" * 80)
    print(f"Evaluating market: {market.upper()}")
    print("=" * 80)

    baseline_test, _ = evaluate_seeded_models(
        HistoricalTradingEnv, test_returns, BASELINE_ENV_KWARGS, baseline_paths
    )
    improved_test, improved_seed_rows = evaluate_seeded_models(
        ImprovedTradingEnv, test_returns, IMPROVED_ENV_KWARGS, improved_paths
    )

    improved_train, _ = evaluate_seeded_models(
        ImprovedTradingEnv, train_returns, IMPROVED_ENV_KWARGS, improved_paths
    )

    random_test = evaluate_strategy(
        HistoricalTradingEnv, test_returns, BASELINE_ENV_KWARGS, "random"
    )
    buyhold_test = evaluate_strategy(
        HistoricalTradingEnv, test_returns, BASELINE_ENV_KWARGS, "buy_and_hold"
    )

    print("\nTest-set results")
    print_row("Improved PPO", improved_test)
    print_row("Baseline PPO", baseline_test)
    print_row("Random", random_test)
    print_row("Buy & Hold", buyhold_test)

    print("\nImproved agent train vs test")
    print_row("Train", improved_train)
    print_row("Test", improved_test)
    print(
        f"Train-Test gap (final P&L mean): "
        f"{improved_train['final_pnl_mean'] - improved_test['final_pnl_mean']:+.2f}"
    )

    results_df = pd.DataFrame(
        [
            {"strategy": "Improved PPO", **{k: v for k, v in improved_test.items() if k != "curve"}},
            {"strategy": "Baseline PPO", **{k: v for k, v in baseline_test.items() if k != "curve"}},
            {"strategy": "Random", **{k: v for k, v in random_test.items() if k != "curve"}},
            {"strategy": "Buy & Hold", **{k: v for k, v in buyhold_test.items() if k != "curve"}},
        ]
    )
    results_csv = RESULTS_DIR / f"{market}_test_results.csv"
    results_df.to_csv(results_csv, index=False)
    print(f"Saved results -> {results_csv}")

    save_four_curve_plot(
        {
            "Improved PPO": improved_test["curve"],
            "Baseline PPO": baseline_test["curve"],
            "Random": random_test["curve"],
            "Buy & Hold": buyhold_test["curve"],
        },
        title=f"{market.upper()} - Test Cumulative P&L",
        filename=PLOTS_DIR / f"{market}_test_cumulative_pnl.png",
    )

    save_train_test_plot(
        improved_train["curve"],
        improved_test["curve"],
        title=f"{market.upper()} - Improved PPO Train vs Test",
        filename=PLOTS_DIR / f"{market}_improved_train_vs_test.png",
    )


if __name__ == "__main__":
    for market in MARKETS:
        evaluate_market(market)

    print("\nFinished evaluation.")