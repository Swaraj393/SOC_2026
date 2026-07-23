import os
import random
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

from baseline_env import HistoricalTradingEnv
from improved_env import ImprovedTradingEnv


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

MARKETS = {
    "spy": DATA_DIR / "spy_returns.npy",
    "nifty": DATA_DIR / "nifty_returns.npy",
}

SEEDS = [0, 1, 2]
TOTAL_TIMESTEPS = 100_000
TRAIN_FRACTION = 0.8

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


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def chronological_split(returns, train_fraction=TRAIN_FRACTION):
    split = int(len(returns) * train_fraction)
    return returns[:split], returns[split:]


def train_one_model(env_cls, train_returns, env_kwargs, market, variant, seed):
    set_seed(seed)

    env = env_cls(train_returns, **env_kwargs)

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        gamma=0.99,
        verbose=1,
        seed=seed,
    )

    print(f"\nTraining {market.upper()} | {variant} | seed={seed}")
    model.learn(total_timesteps=TOTAL_TIMESTEPS)

    model_path = MODELS_DIR / f"clean_{market}_{variant}_seed_{seed}"
    model.save(str(model_path))
    print(f"Saved model to {model_path}.zip")


def train_market(market, returns):
    train_returns, test_returns = chronological_split(returns)

    print("\n" + "=" * 70)
    print(f"MARKET: {market.upper()}")
    print(f"Train days: {len(train_returns)} | Test days: {len(test_returns)}")
    print("=" * 70)

    for seed in SEEDS:
        train_one_model(
            HistoricalTradingEnv,
            train_returns,
            BASELINE_ENV_KWARGS,
            market,
            "baseline",
            seed,
        )

    for seed in SEEDS:
        train_one_model(
            ImprovedTradingEnv,
            train_returns,
            IMPROVED_ENV_KWARGS,
            market,
            "improved",
            seed,
        )


if __name__ == "__main__":
    os.makedirs(MODELS_DIR, exist_ok=True)

    for market, path in MARKETS.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing data file: {path}")
        returns = np.load(path)
        train_market(market, returns)

    print("\nFinished training all clean models.")